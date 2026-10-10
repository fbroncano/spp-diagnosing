"""Downloads the reference series of the Nevada Geodetic Laboratory (NGL).

For each station it reads the daily PPP series in IGS20 (`.tenv3`) and saves it
to `data/ngl/<STATION>.csv` with the date and the ECEF position in metres. The
data are a third-party product: they are downloaded, not redistributed with the
code.

Usage:
    python -m analysis.reference [STATION ...]

Without arguments it downloads the 27 stations of the study.
"""

from pathlib import Path

import pandas as pd
from pyproj import Transformer

from analysis._paths import REFERENCE_DIR

BASE_URL = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{station}.tenv3"
DATA_DIR = REFERENCE_DIR

# IGS20 geographic 3D (lat, lon, ellipsoidal height) -> IGS20 geocentric / ECEF (X, Y, Z).
# EPSG:10178 is the 2D variant of IGS20 and would discard the height.
_TO_ECEF = Transformer.from_crs("EPSG:10177", "EPSG:10176", always_xy=True)


def process_station(station: str) -> Path:
    """
    Downloads the .tenv3 time series of a station and saves a CSV in data/ngl
    with the measurement date and the final ECEF position (EPSG:4978, metres).

    Returns the path of the generated CSV.
    """
    station = station.upper()
    df = pd.read_csv(BASE_URL.format(station=station), sep=r"\s+")

    # YYMMMDD (e.g. 05NOV17); the NGL series start in the 1990s
    date = pd.to_datetime(df["YYMMMDD"], format="%y%b%d")
    date = date.mask(date > pd.Timestamp.now(), date - pd.DateOffset(years=100))

    x, y, z = _TO_ECEF.transform(
        df["_longitude(deg)"].to_numpy(),
        df["_latitude(deg)"].to_numpy(),
        df["__height(m)"].to_numpy(),
    )

    out = pd.DataFrame({"date": date.dt.date, "x": x, "y": y, "z": z})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{station}.csv"
    out.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import sys

    from analysis.dataset import ESTACIONES

    stations = sys.argv[1:] or ESTACIONES

    for station in stations:
        print(process_station(station))
