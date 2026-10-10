"""Characterisation of the SPP processing: estimation methods x constellations.

For each (constellation, method) combination, the quality of the daily position
is measured against the NGL PPP series (Table 2 of the article).

Three things that are often conflated in a single figure are kept apart:

  accuracy    distance to the true position. It includes the systematic bias
              of the station (instrumental delay, ARP error, tropospheric
              model), common to every day.
  precision   day-to-day repeatability, computed after removing from each
              station its own central value. This is what really separates
              the estimators: they share the same bias but reject noise
              differently.
  bias        signed mean per component. In the vertical, it reveals
              processing artefacts.

A robust estimator cannot improve accuracy if the bias is systematic: it can
only improve precision. Reporting the 3D RMS alone hides this.

Two caveats on the comparison between constellations:

  1. `GEC` is the processing mode requested, not the constellations that
     actually entered. It is truly multi-constellation only when the station's
     navigation file carries BeiDou on that day; otherwise the solution is GE
     labelled GEC. These rows are filtered with `n_sat_c_avg > 0`.

     The cause is **not** the IGN migration: on 1 October 2023 IGN stopped
     publishing per-constellation files (`GN`, `EN`, `RN`) and moved to the
     combined `MN` file, and the study period starts on that day, so every
     station has `MN` from the first row. The migration only sets the lower
     bound of the period.

     The actual cause is **the receiver**: 16 of the 27 stations have days
     without BeiDou ephemerides and four (RIO1, MALA, BCLN, TAR0) never publish
     them. The affected rows are spread evenly over the three years. It is the
     same phenomenon as at CACE, which stopped outputting BeiDou when its
     receiver was replaced on 2025-03-19.
  2. BeiDou alone is not available at every station. With `--comun` the table
     is restricted to the (station, day) pairs present in all 20 combinations,
     the only strictly paired comparison.
"""
import argparse

import numpy as np
import pandas as pd

from analysis.dataset import (CONSTELACIONES, DESDE, HASTA, ESTACIONES, METODOS,
                        TARGETS, build_dataset)

# Presentation order: from fewer to more constellations, and the estimators
# grouped by family (simple, robust, weighted).
ORDEN_CONST = ["G", "E", "C", "GEC"]
ORDEN_METODO = ["mean", "median", "mad", "weighted", "weighted_mad"]


def cargar_todo(stations=None, desde=DESDE, hasta=HASTA, db=None):
    """Full dataset, with no constellation or method fixed."""
    df = build_dataset(stations or ESTACIONES, db=db)
    df = df[(df["date"] >= pd.Timestamp(desde)) & (df["date"] <= pd.Timestamp(hasta))]
    return df.reset_index(drop=True)


def marcar_gec_real(df):
    """Discards the `GEC` rows that are actually GE (no BeiDou in the navigation file)."""
    es_gec = df["constellation"] == "GEC"
    sin_beidou = df["n_sat_c_avg"].fillna(0) <= 0
    descartadas = int((es_gec & sin_beidou).sum())
    if descartadas:
        print(f"  GEC without BeiDou in the navigation file: {descartadas} rows discarded "
              f"(GE labelled as GEC)")
    return df[~(es_gec & sin_beidou)].reset_index(drop=True)


def subconjunto_comun(df):
    """(station, day) pairs present in all 20 combinations.

    This is the paired comparison: without it, BeiDou is measured over 23
    stations and the rest over 27, so differences between constellations are
    mixed with differences between stations.
    """
    n_combos = df.groupby(["constellation", "method"], observed=True).ngroups
    clave = df.groupby(["station_id", "date"]).size()
    completos = clave[clave == n_combos].index
    idx = pd.MultiIndex.from_frame(df[["station_id", "date"]])
    return df[idx.isin(completos)].reset_index(drop=True)


def _precision(sub):
    """Repeatability: dispersion after removing each station's central value.

    Without centring per station, the dispersion would mostly measure how
    different the stations are from each other, not how stable the estimator
    is. It is the same care that Simpson's paradox demands in hierarchical data.

    Centring uses the median and dispersion the scaled MAD, not mean and
    standard deviation: `mean` and `weighted` produce days with kilometre-scale
    errors, and a single such solution dominates any variance. With the MAD the
    figure still describes 99% of the days.
    """
    centrado = sub.groupby("station_id", observed=True)[TARGETS].transform(
        lambda s: s - s.median())
    mad = 1.4826 * centrado.abs().median()
    return {
        "prec_e": float(mad["d_e"]),
        "prec_n": float(mad["d_n"]),
        "prec_u": float(mad["d_u"]),
        "prec_3d": float(np.sqrt((mad ** 2).sum())),
    }


def resumir(sub):
    """Metrics of one (constellation, method) combination."""
    enu = sub[TARGETS].to_numpy()
    e3d = np.linalg.norm(enu, axis=1)
    e2d = np.linalg.norm(enu[:, :2], axis=1)
    fila = {
        "n": len(sub),
        "est": sub["station_id"].nunique(),
        "dias": sub["date"].nunique(),
        # Accuracy: includes the systematic bias.
        "rmse_3d": float(np.sqrt(np.mean(e3d ** 2))),
        "mae_3d": float(np.mean(e3d)),
        "p95_3d": float(np.percentile(e3d, 95)),
        "rmse_2d": float(np.sqrt(np.mean(e2d ** 2))),
        "rmse_u": float(np.sqrt(np.mean(enu[:, 2] ** 2))),
        # Signed bias (mean), per component.
        "sesgo_e": float(enu[:, 0].mean()),
        "sesgo_n": float(enu[:, 1].mean()),
        "sesgo_u": float(enu[:, 2].mean()),
    }
    fila.update(_precision(sub))

    # Robustness. The estimators without outlier rejection (`mean`, `weighted`)
    # are not biased: their median is normal, but they ruin a small fraction of
    # days with kilometre-scale errors. Over that tail, mean and RMSE stop
    # meaning anything, so estimators must be compared with the median and the
    # failure rate.
    fila["mediana_3d"] = float(np.median(e3d))
    fila["p99_3d"] = float(np.percentile(e3d, 99))
    fila["max_3d"] = float(e3d.max())
    fila["fallo_10m_%"] = float(100 * (e3d > 10).mean())
    fila["fallo_100m_%"] = float(100 * (e3d > 100).mean())

    # Availability: share of the day that survives epoch filtering.
    fila["epocas_%"] = float(100 * sub["epoch_ratio"].mean())
    fila["n_sat"] = float(sub["n_satellites_avg"].mean())
    return fila


def tabla(df):
    """Full constellation x method table."""
    filas = {}
    for (const, metodo), sub in df.groupby(["constellation", "method"], observed=True):
        filas[(const, metodo)] = resumir(sub)
    out = pd.DataFrame(filas).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=["constelacion", "metodo"])
    const = [c for c in ORDEN_CONST if c in out.index.get_level_values(0)]
    met = [m for m in ORDEN_METODO if m in out.index.get_level_values(1)]
    return out.reindex(pd.MultiIndex.from_product([const, met],
                                                  names=["constelacion", "metodo"])).dropna(how="all")


def pivote(out, metrica):
    """Constellation x method view of a single metric."""
    return out[metrica].unstack("metodo").reindex(
        columns=[m for m in ORDEN_METODO if m in out.index.get_level_values(1)])


def por_estacion(df, metrica="err_3d"):
    """Median error per station and constellation.

    Median rather than mean, for consistency with the rest of the analysis: the
    mean is dominated by the tail, which is precisely what must be separated
    from the typical case here.
    """
    return (df.groupby(["constellation", "station_id"], observed=True)[metrica]
              .median().unstack("constellation"))


def por_antena(df, metrica="err_3d"):
    """Median error per antenna model and constellation, with the E/G ratio (Table 3).

    The Galileo/GPS ratio is what isolates the effect: a site, multipath or
    misapplied-offset problem would degrade both constellations at once, so a
    high E/G with GPS intact points to a frequency-selective cause. The
    Galileo ionosphere-free combination uses E5a (1176 MHz), outside the design
    band of 1990s L1/L2 GPS antennas. `receptor.py` checks the receiver as the
    alternative frequency-selective explanation.
    """
    out = df.pivot_table(index="antenna_type", columns="constellation",
                         values=metrica, aggfunc="median", observed=True)
    if "E" in out.columns and "G" in out.columns:
        out["E/G"] = out["E"] / out["G"]
    out["n_est"] = df.groupby("antenna_type", observed=True)["station_id"].nunique()
    out["estaciones"] = df.groupby("antenna_type", observed=True)["station_id"].agg(
        lambda s: " ".join(sorted(s.unique())))
    return out.sort_values("E/G", ascending=False) if "E/G" in out else out


def main():
    parser = argparse.ArgumentParser(
        description="SPP characterisation: estimation methods x constellations")
    parser.add_argument("--stations", nargs="+", default=None)
    parser.add_argument("--desde", default=DESDE)
    parser.add_argument("--hasta", default=HASTA)
    parser.add_argument("--comun", action="store_true",
                        help="restrict to the (station, day) pairs present in all 20 combinations")
    parser.add_argument("--salida", default=None, help="prefix of the output CSV files")
    args = parser.parse_args()

    df = cargar_todo(args.stations, args.desde, args.hasta)
    print(f"Raw: {len(df)} solutions | {df['station_id'].nunique()} stations | "
          f"{df['date'].nunique()} days ({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d})")

    df = marcar_gec_real(df)
    if args.comun:
        antes = len(df)
        df = subconjunto_comun(df)
        print(f"  paired common subset: {len(df)} of {antes} rows")
    print(f"Analysed: {len(df)} solutions in "
          f"{df.groupby(['constellation', 'method'], observed=True).ngroups} combinations\n")

    out = tabla(df)

    print("=== Full table (metres) ===")
    print(out.round(3).to_string(), "\n")

    for etiqueta, metrica in [("Accuracy: median 3D error", "mediana_3d"),
                              ("Tail: 99th percentile of the 3D error", "p99_3d"),
                              ("Reliability: % of days with error > 10 m", "fallo_10m_%"),
                              ("Precision: 3D repeatability (MAD centred per station)", "prec_3d"),
                              ("Vertical precision", "prec_u"),
                              ("Vertical bias (signed mean)", "sesgo_u")]:
        print(f"=== {etiqueta} ===")
        print(pivote(out, metrica).round(3).to_string(), "\n")

    mejor = df[df["method"] == "weighted_mad"]

    print("=== MAIN RESULT: MEDIAN 3D error per antenna model "
          "(weighted_mad) ===")
    print(por_antena(mejor).round(3).to_string(), "\n")

    print("=== MEDIAN 3D error per station and constellation (weighted_mad) ===")
    print(por_estacion(mejor).round(3).to_string(), "\n")

    if args.salida:
        out.to_csv(f"{args.salida}_completa.csv")
        for metrica in ["mediana_3d", "p99_3d", "fallo_10m_%", "prec_3d", "prec_u", "sesgo_u"]:
            pivote(out, metrica).to_csv(f"{args.salida}_{metrica}.csv")
        por_estacion(mejor).to_csv(f"{args.salida}_por_estacion.csv")
        por_antena(mejor).to_csv(f"{args.salida}_por_antena.csv")
        print(f"tables saved with prefix {args.salida}_")


if __name__ == "__main__":
    main()
