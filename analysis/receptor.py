"""Separates the effect of the antenna from that of the receiver.

The invariance of GPS is not enough to rule out the receiver (Section 4.2 of
the article): signal tracking is band specific, and a receiver may deliver
clean L1/L2 and deficient E5a. Since the receiver metadata are already in the
database, the check is direct.

Three questions, in this order:

1. Does the structure follow the receiver as well as the antenna?
2. Are antenna and receiver confounded, i.e. does each antenna model always
   come with the same receiver? If so, no observational data can separate
   them, and that must be stated.
3. Within one receiver model, does E/G still vary with the antenna? And the
   other way round. This is the decisive question.

It also computes bootstrap intervals over stations and permutation tests,
which replace correlations between ratios and their components (spurious by
construction).

Usage:  python -m analysis.receptor [--salida results/receptor]
"""

import argparse

import numpy as np
import pandas as pd

from analysis.caracterizacion import cargar_todo, marcar_gec_real, por_antena
from analysis.dataset import DESDE, HASTA

SEMILLA = 20260921
N_BOOT = 10_000
N_PERM = 10_000


def _mediana_por_estacion(sub):
    """Median 3D error per station and constellation.

    The independent unit is the station, not the day: the 1 001 days of a
    station share antenna, site and receiver, so treating them as independent
    observations inflates any test.
    """
    return (sub.groupby(["station_id", "constellation"], observed=True)["err_3d"]
               .median().unstack("constellation"))


def por_receptor(mejor):
    """Table 3 of the article, grouped by receiver model instead of antenna."""
    out = mejor.pivot_table(index="receiver_type", columns="constellation",
                            values="err_3d", aggfunc="median", observed=True)
    if "E" in out.columns and "G" in out.columns:
        out["E/G"] = out["E"] / out["G"]
    out["n_est"] = mejor.groupby("receiver_type", observed=True)["station_id"].nunique()
    out["estaciones"] = mejor.groupby("receiver_type", observed=True)["station_id"].agg(
        lambda s: " ".join(sorted(s.unique())))
    return out.sort_values("E/G", ascending=False) if "E/G" in out else out


def por_antena_y_receptor(mejor):
    """Cross of both metadata: one row per (antenna, receiver) pair."""
    out = mejor.pivot_table(index=["antenna_type", "receiver_type"],
                            columns="constellation", values="err_3d",
                            aggfunc="median", observed=True)
    if "E" in out.columns and "G" in out.columns:
        out["E/G"] = out["E"] / out["G"]
    out["n_est"] = mejor.groupby(["antenna_type", "receiver_type"],
                                 observed=True)["station_id"].nunique()
    out["estaciones"] = mejor.groupby(["antenna_type", "receiver_type"],
                                      observed=True)["station_id"].agg(
        lambda s: " ".join(sorted(s.unique())))
    return out.sort_values("E/G", ascending=False) if "E/G" in out else out


def confusion(mejor):
    """Can the design separate antenna from receiver?

    Counts, per station, the (antenna, receiver) pair in force. If each antenna
    model appears with a single receiver and vice versa, the two variables are
    indistinguishable with these data and no analysis can fix it.
    """
    est = (mejor.groupby("station_id", observed=True)[["antenna_type", "receiver_type"]]
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None))
    tabla = pd.crosstab(est["antenna_type"], est["receiver_type"])
    rec_por_ant = est.groupby("antenna_type")["receiver_type"].nunique()
    ant_por_rec = est.groupby("receiver_type")["antenna_type"].nunique()
    return tabla, rec_por_ant, ant_por_rec, est


def eg_por_estacion(mejor):
    """E/G ratio per station and instrument period.

    The unit is NOT the station but the (station, instrument period) pair: CACE
    replaced its TRM29659.00 with a LEIAR20 on 19 March 2025, so collapsing it
    into a single record would assign it an E/G averaged over both periods and
    contaminate precisely the row that carries the result. With this split
    CACE contributes one observation to each model.
    """
    med = (mejor.groupby(["station_id", "antenna_type", "receiver_type",
                          "constellation"], observed=True)["err_3d"]
                .median().unstack("constellation"))
    med = med[["G", "E"]].dropna()
    med["E/G"] = med["E"] / med["G"]
    return med.reset_index().set_index("station_id")


def bootstrap_eg(eg, por="antenna_type", n=N_BOOT, semilla=SEMILLA):
    """95 % interval of the E/G ratio of each group, resampling stations.

    Resampling is over stations because the station is the replicated unit.
    Groups with a single station admit no interval and are flagged as such
    (see Section 4.2 of the article).
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for grupo, sub in eg.groupby(por, observed=True):
        g, e = sub["G"].to_numpy(), sub["E"].to_numpy()
        k = len(sub)
        punto = np.median(e) / np.median(g)
        if k < 2:
            filas.append({por: grupo, "n_est": k, "E/G": punto,
                          "ic_bajo": np.nan, "ic_alto": np.nan})
            continue
        idx = rng.integers(0, k, size=(n, k))
        reps = np.median(e[idx], axis=1) / np.median(g[idx], axis=1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        filas.append({por: grupo, "n_est": k, "E/G": punto,
                      "ic_bajo": lo, "ic_alto": hi})
    return pd.DataFrame(filas).sort_values("E/G", ascending=False)


def permutacion(eg, por="antenna_type", n=N_PERM, semilla=SEMILLA):
    """Does the metadatum explain the dispersion of E/G better than chance?

    Replaces the correlations of log(E/G) with log(E) and log(G), which are
    spurious by construction (Pearson, 1897): log(E/G) = log E - log G, so its
    correlation with each component has a sign fixed in advance.

    Two statistics are computed on log(E/G), because they measure different
    things:

    - `eta2`, the between-group fraction of variance. It weights by group size,
      so a single-station group cannot dominate it.
    - `recorrido` (range), the difference between the largest and smallest
      group median. It describes the effect, but as a test statistic it has
      almost no power here: with nine groups of sizes 1 to 9, a permutation
      that sends an extreme value to a single-station group reproduces the
      observed range effortlessly.

    The null hypothesis reassigns the labels across units while preserving
    group sizes.
    """
    rng = np.random.default_rng(semilla)
    sub = eg.dropna(subset=[por]).copy()
    valores = np.log(sub["E/G"].to_numpy())
    etiquetas = sub[por].to_numpy()

    def estadisticos(lab):
        s = pd.Series(valores)
        med = s.groupby(lab).median()
        gran = valores.mean()
        medias = s.groupby(lab).mean()
        tam = s.groupby(lab).size()
        entre = float((tam * (medias - gran) ** 2).sum())
        total = float(((valores - gran) ** 2).sum())
        return entre / total if total else np.nan, float(med.max() - med.min())

    obs = estadisticos(etiquetas)
    nulo = np.empty((n, 2))
    for i in range(n):
        nulo[i] = estadisticos(rng.permutation(etiquetas))
    p = [(1 + np.sum(nulo[:, j] >= obs[j])) / (n + 1) for j in range(2)]
    return obs, p, nulo


# Legacy antenna models of GPS L1/L2 design, predating E5a, as classified in the
# article. LEIAT504 (ACOR), of the same family and period as LEIAT504GG, is not
# included; the article discusses that case (Section 4.2).
HEREDADAS = {"LEIAT504GG", "TRM29659.00", "LEIAR25"}


def contraste_heredadas(eg, n=N_PERM, semilla=SEMILLA):
    """Single-hypothesis test: legacy versus modern antenna models.

    This is the test with power, because it spends one degree of freedom
    instead of eight. The classification is by design period of the antenna
    model, an attribute of the instrument that precedes the data, not a cut
    chosen after looking at the residuals.
    """
    rng = np.random.default_rng(semilla)
    sub = eg.dropna(subset=["antenna_type"]).copy()
    es_her = sub["antenna_type"].isin(HEREDADAS).to_numpy()
    valores = np.log(sub["E/G"].to_numpy())

    def estadistico(mascara):
        return float(np.median(valores[mascara]) - np.median(valores[~mascara]))

    obs = estadistico(es_her)
    nulo = np.empty(n)
    for i in range(n):
        nulo[i] = estadistico(rng.permutation(es_her))
    p = (1 + np.sum(np.abs(nulo) >= abs(obs))) / (n + 1)
    return obs, p, int(es_her.sum()), int((~es_her).sum())


def main():
    parser = argparse.ArgumentParser(
        description="Antenna versus receiver in the Galileo degradation")
    parser.add_argument("--stations", nargs="+", default=None)
    parser.add_argument("--desde", default=DESDE)
    parser.add_argument("--hasta", default=HASTA)
    parser.add_argument("--salida", default=None, help="prefix of the output CSV files")
    args = parser.parse_args()

    df = cargar_todo(args.stations, args.desde, args.hasta)
    df = marcar_gec_real(df)
    mejor = df[df["method"] == "weighted_mad"]
    print(f"weighted_mad: {len(mejor)} solutions | "
          f"{mejor['station_id'].nunique()} stations\n")

    print("=== Reference: median 3D error per ANTENNA (Table 3 of the article) ===")
    t_ant = por_antena(mejor)
    print(t_ant.round(3).to_string(), "\n")

    print("=== The same table per RECEIVER ===")
    t_rec = por_receptor(mejor)
    print(t_rec.round(3).to_string(), "\n")

    print("=== Are antenna and receiver confounded? ===")
    cruce, rec_por_ant, ant_por_rec, _ = confusion(mejor)
    print(cruce.to_string(), "\n")
    print("Distinct receivers per antenna model:")
    print(rec_por_ant.to_string(), "\n")
    print("Distinct antennas per receiver model:")
    print(ant_por_rec.to_string(), "\n")

    print("=== Cross (antenna, receiver) ===")
    t_cruce = por_antena_y_receptor(mejor)
    print(t_cruce.round(3).to_string(), "\n")

    eg = eg_por_estacion(mejor)
    print("=== E/G per station, with its equipment ===")
    print(eg.sort_values("E/G", ascending=False).round(3).to_string(), "\n")

    print("=== 95 % intervals by resampling stations ===")
    ic_ant = bootstrap_eg(eg, "antenna_type")
    print("Per antenna:")
    print(ic_ant.round(3).to_string(index=False), "\n")
    ic_rec = bootstrap_eg(eg, "receiver_type")
    print("Per receiver:")
    print(ic_rec.round(3).to_string(index=False), "\n")

    print("=== Permutation tests ===")
    for por in ["antenna_type", "receiver_type"]:
        (eta2, rec), (p_eta, p_rec), _ = permutacion(eg, por)
        print(f"  {por:15s}  eta2 = {eta2:.3f} (p = {p_eta:.4f})   "
              f"range = {rec:.3f} (p = {p_rec:.4f})")
    obs, p, n_her, n_mod = contraste_heredadas(eg)
    print(f"\n  legacy ({n_her}) vs modern ({n_mod}): "
          f"difference of medians of log(E/G) = {obs:.3f} "
          f"(ratio {np.exp(obs):.2f}x)  p = {p:.4f}")
    print()

    if args.salida:
        t_rec.to_csv(f"{args.salida}_por_receptor.csv")
        t_cruce.to_csv(f"{args.salida}_antena_receptor.csv")
        cruce.to_csv(f"{args.salida}_confusion.csv")
        eg.to_csv(f"{args.salida}_eg_por_estacion.csv")
        ic_ant.to_csv(f"{args.salida}_ic_antena.csv", index=False)
        ic_rec.to_csv(f"{args.salida}_ic_receptor.csv", index=False)
        print(f"tables saved with prefix {args.salida}_")


if __name__ == "__main__":
    main()
