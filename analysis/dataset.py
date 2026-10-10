"""Joins the SPP solutions in the database with the NGL reference and computes the error.

Each row is a daily SPP solution (station, date, constellation, method) with
its quality statistics, the equipment in force on that day and the error
against the NGL PPP series, rotated into the local ENU frame:

    d = ENU(PPP - SPP)        err_3d = |d|

The SPP solution refers to the antenna phase centre and the NGL one to the
monument mark, so the DELTA H in force (table `station_antenna`, loaded from the
EPN site logs) is added to the vertical component, projected onto the local
vertical and not onto Z.

Frames: the SPP solutions are in WGS84 (realised by the broadcast ephemerides)
and the NGL reference in IGS20. Both agree at the ~1 cm level, two orders of
magnitude below the error evaluated. Plate motion needs no epoch reduction
because each SPP solution is compared with the PPP position of the same day.
"""
import os

import numpy as np
import pandas as pd

from analysis._paths import REFERENCE_DIR
from analysis.geodesy import ecef2lla, ecef_delta_to_enu
from dao import Database

_QUERY = """
SELECT dp.station_id,
       dp.date,
       dp.constellation_id,
       dp.method_id,
       c.code  AS constellation,
       m.code  AS method,
       ST_X(dp.position) AS spp_x,
       ST_Y(dp.position) AS spp_y,
       ST_Z(dp.position) AS spp_z,
       dp.std_x, dp.std_y, dp.std_z, dp.std_3d,
       dp.n_epochs_used, dp.n_epochs,
       ds.n_satellites_avg, ds.clock_bias_m,
       ds.gdop_avg, ds.pdop_avg, ds.pdop_max, ds.tdop_avg,
       ds.rms_avg, ds.rms_max,
       ds.gdop_p95, ds.pdop_p95, ds.tdop_p95, ds.rms_p95,
       ds.hdop_avg, ds.hdop_p95, ds.hdop_max,
       ds.vdop_avg, ds.vdop_p95, ds.vdop_max,
       ds.elev_avg, ds.elev_min, ds.n_sat_low_avg,
       ds.azim_resultant, ds.azim_direction,
       ds.n_sat_g_avg, ds.n_sat_e_avg, ds.n_sat_c_avg,
       -- Equipment in force on that day. The interval is half-open so that
       -- the day of a change falls in the new record.
       (SELECT sa.delta_h FROM station_antenna sa
         WHERE sa.station_id = dp.station_id
           AND dp.date >= sa.valid_from::date
           AND (sa.valid_to IS NULL OR dp.date < sa.valid_to::date)
         LIMIT 1) AS delta_h,
       (SELECT sa.antenna_type FROM station_antenna sa
         WHERE sa.station_id = dp.station_id
           AND dp.date >= sa.valid_from::date
           AND (sa.valid_to IS NULL OR dp.date < sa.valid_to::date)
         LIMIT 1) AS antenna_type,
       (SELECT sr.receiver_type FROM station_receiver sr
         WHERE sr.station_id = dp.station_id
           AND dp.date >= sr.valid_from::date
           AND (sr.valid_to IS NULL OR dp.date < sr.valid_to::date)
         LIMIT 1) AS receiver_type
FROM daily_positions dp
JOIN constellations c ON c.id = dp.constellation_id
JOIN methods       m ON m.id = dp.method_id
LEFT JOIN daily_statistics ds
       ON ds.station_id = dp.station_id
      AND ds.date = dp.date
      AND ds.constellation_id = dp.constellation_id
WHERE (%(stations)s IS NULL OR dp.station_id = ANY(%(stations)s))
  -- `hdop_avg` is filled only by the corrected pipeline (ARP reduction and one
  -- ionosphere-free combination per constellation). It marks those rows so
  -- that they are not mixed with older rows, which carry a ~0.7 m vertical
  -- bias in GPS.
  AND (NOT %(solo_nuevo)s OR ds.hdop_avg IS NOT NULL)
ORDER BY dp.station_id, dp.date, dp.constellation_id, dp.method_id
"""

# Numeric quality columns of the SPP solution.
QUALITY_COLS = [
    "std_x", "std_y", "std_z", "std_3d",
    "n_epochs_used", "n_epochs",
    "n_satellites_avg", "clock_bias_m",
    "gdop_avg", "pdop_avg", "pdop_max", "tdop_avg",
    "rms_avg", "rms_max",
    # Added with the corrected pipeline. They exist only in its rows, which are
    # identified by a non-null `hdop_avg`.
    "gdop_p95", "pdop_p95", "tdop_p95", "rms_p95",
    "hdop_avg", "hdop_p95", "hdop_max",
    "vdop_avg", "vdop_p95", "vdop_max",
    "elev_avg", "elev_min", "n_sat_low_avg",
    "n_sat_g_avg", "n_sat_e_avg", "n_sat_c_avg",
]

CATEGORICAL_COLS = ["constellation", "method"]

TARGETS = ["d_e", "d_n", "d_u"]

# ---------------------------------------------------------------------------
# Study configuration.
# ---------------------------------------------------------------------------

# Spanish EPN stations, except VALA, which never publishes a navigation file
# (the study uses only each station's own data).
ESTACIONES = [
    "ACOR", "ALAC", "ALBA", "ALME", "BCLN", "CACE", "CANT", "CARG", "CEU1",
    "COBA", "HUEL", "IBIZ", "IZAN", "LEON", "LPAL", "MALA", "MALL",
    "MELI", "RIO1", "SALA", "SONS", "TAR0", "TERU", "VALE", "VIGO", "YEBE",
    "ZARA",
]

# The lower bound is the IGN migration to multi-constellation navigation files
# (`MN`): there is no BeiDou before it. The upper bound is set by the NGL series.
DESDE = "2023-10-01"
HASTA = "2026-06-30"

CONSTELACIONES = ["G", "E", "C", "GEC"]
METODOS = ["mean", "median", "mad", "weighted", "weighted_mad"]


def load_spp(stations=None, db=None, solo_nuevo=True):
    """Reads the SPP solutions and their statistics from PostGIS."""
    db = db or Database()
    with db.get_cursor() as cursor:
        cursor.execute(_QUERY, {"stations": list(stations) if stations else None,
                                "solo_nuevo": solo_nuevo})
        df = pd.DataFrame([dict(r) for r in cursor.fetchall()])
    df["date"] = pd.to_datetime(df["date"])
    for col in QUALITY_COLS + ["spp_x", "spp_y", "spp_z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_truth(station, folder=REFERENCE_DIR):
    """Daily NGL PPP series of a station (data/ngl/<STATION>.csv)."""
    path = os.path.join(folder, f"{station}.csv")
    truth = pd.read_csv(path, parse_dates=["date"])
    truth = truth.rename(columns={"x": "ppp_x", "y": "ppp_y", "z": "ppp_z"})
    truth["station_id"] = station
    return truth[["station_id", "date", "ppp_x", "ppp_y", "ppp_z"]]


def station_origin(station, db=None):
    """Reference (lat, lon) of the station, used only to define the ENU frame."""
    db = db or Database()
    with db.get_cursor() as cursor:
        cursor.execute(
            "SELECT ST_X(location) x, ST_Y(location) y, ST_Z(location) z "
            "FROM stations WHERE id = %s", (station,))
        row = cursor.fetchone()
    lat, lon, _ = ecef2lla(row["x"], row["y"], row["z"])
    return lat, lon


def build_dataset(stations, db=None, folder=REFERENCE_DIR, solo_nuevo=True):
    """Joins SPP + PPP reference and computes the ENU error of each solution."""
    db = db or Database()
    stations = [stations] if isinstance(stations, str) else list(stations)

    spp = load_spp(stations, db=db, solo_nuevo=solo_nuevo)
    truth = pd.concat([load_truth(s, folder) for s in stations], ignore_index=True)
    df = spp.merge(truth, on=["station_id", "date"], how="inner")

    # ECEF -> ENU error in the local frame of each station.
    enu = np.zeros((len(df), 3))
    for station, idx in df.groupby("station_id").groups.items():
        lat, lon = station_origin(station, db=db)
        rows = df.loc[idx]
        delta = rows[["ppp_x", "ppp_y", "ppp_z"]].to_numpy() - rows[["spp_x", "spp_y", "spp_z"]].to_numpy()
        enu[df.index.get_indexer(idx)] = ecef_delta_to_enu(delta, lat, lon)
    df[TARGETS] = enu

    # Reduction to the antenna reference point: the SPP solution lies DELTA H
    # above the mark, so the vertical residual is shifted by +DELTA H. The value
    # varies with the date because stations change antennas.
    df["delta_h"] = pd.to_numeric(df["delta_h"], errors="coerce")
    df["delta_h_known"] = df["delta_h"].notna()
    df["d_u"] = df["d_u"] + df["delta_h"].fillna(0.0)

    df["epoch_ratio"] = df["n_epochs_used"] / df["n_epochs"].replace(0, np.nan)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    df["err_3d"] = np.linalg.norm(df[TARGETS].to_numpy(), axis=1)
    df["err_2d"] = np.linalg.norm(df[["d_e", "d_n"]].to_numpy(), axis=1)
    return df.reset_index(drop=True)
