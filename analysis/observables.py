"""Which observable does the processor resolve in each band, and at what cost?

`_resolve_code` in `pre_processor.py` resolves a band prefix to the FIRST
header column that starts with it, with no criterion on the tracking attribute.
For GPS this matters: the broadcast clock refers to the L1/L2 ionosphere-free
combination on the P(Y) code (`C2W`), so a station that declares `C2S` (the
civil L2C) before `C2W` resolves the band to the wrong code and carries the
differential bias between the two.

This script reads the RINEX observation headers, reproduces the resolution made
by the processor and crosses it with the observed error. If the header order
predicts a step in the GPS error, there is a live artefact.

Usage:  python -m analysis.observables --cabeceras <dir> [--salida results/observables]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.caracterizacion import cargar_todo, marcar_gec_real
from analysis.dataset import DESDE, HASTA

# The pairs as declared in the processor CONSTANTS.py.
PREFIJOS = {"G": ("C1", "C2"), "E": ("C1", "C5")}


def leer_cabecera(ruta):
    """Antenna, radome, receiver, firmware and codes per system."""
    texto = ruta.read_text(errors="replace")
    datos = {"station_id": ruta.stem}

    for linea in texto.splitlines():
        if "ANT # / TYPE" in linea:
            campo = linea[20:60]
            datos["antenna_type"] = campo[:16].strip()
            datos["radomo"] = campo[16:].strip() or "NONE"
        elif "REC # / TYPE / VERS" in linea:
            datos["receiver_type"] = linea[20:40].strip()
            datos["firmware"] = linea[40:60].strip()

    # Observable lists may continue on lines without a system letter.
    codigos, sistema = {}, None
    for linea in texto.splitlines():
        if "SYS / # / OBS TYPES" not in linea:
            continue
        cabeza = linea[:60]
        if re.match(r"^[GRECJIS] ", cabeza):
            sistema = cabeza[0]
            codigos.setdefault(sistema, [])
            cuerpo = cabeza[6:]
        else:
            cuerpo = cabeza[6:]
        if sistema:
            codigos[sistema].extend(cuerpo.split())

    for s, lista in codigos.items():
        datos[f"obs_{s}"] = " ".join(c for c in lista if c.startswith("C"))
    return datos


def resolver(codigos, prefijo):
    """Reproduces `_resolve_code`: the first column matching the prefix."""
    for c in codigos.split():
        if c.startswith(prefijo):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Observable resolved per band and its effect on the error")
    parser.add_argument("--cabeceras", required=True, help="directory with one RINEX header per station, <STATION>.txt")
    parser.add_argument("--desde", default=DESDE)
    parser.add_argument("--hasta", default=HASTA)
    parser.add_argument("--salida", default=None)
    args = parser.parse_args()

    filas = [leer_cabecera(p) for p in sorted(Path(args.cabeceras).glob("*.txt"))
             if p.stat().st_size]
    cab = pd.DataFrame(filas).set_index("station_id")

    for sistema, (p1, p2) in PREFIJOS.items():
        col = f"obs_{sistema}"
        if col not in cab:
            continue
        cab[f"{sistema}1"] = cab[col].fillna("").map(lambda c: resolver(c, p1))
        cab[f"{sistema}2"] = cab[col].fillna("").map(lambda c: resolver(c, p2))

    df = cargar_todo(None, args.desde, args.hasta)
    df = marcar_gec_real(df)
    mejor = df[df["method"] == "weighted_mad"]
    err = (mejor.groupby(["station_id", "constellation"], observed=True)["err_3d"]
                .median().unstack("constellation"))

    tabla = cab.join(err[["G", "E"]], how="left")
    tabla["E/G"] = tabla["E"] / tabla["G"]

    print("=== Observable resolved per station ===")
    cols = ["antenna_type", "radomo", "receiver_type", "G1", "G2", "E1", "E2", "G", "E", "E/G"]
    print(tabla[[c for c in cols if c in tabla]].sort_values("G2").round(3).to_string(), "\n")

    print("=== GPS error by the code resolving the L2 band ===")
    grupo = tabla.groupby("G2")["G"].agg(["count", "median", "min", "max"])
    print(grupo.round(3).to_string(), "\n")

    c2s = tabla[tabla["G2"] == "C2S"]["G"].dropna()
    c2w = tabla[tabla["G2"] == "C2W"]["G"].dropna()
    if len(c2s) >= 2 and len(c2w) >= 2:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(c2s, c2w)
        print(f"  C2S n={len(c2s)} median {np.median(c2s):.3f} m | "
              f"C2W n={len(c2w)} median {np.median(c2w):.3f} m")
        print(f"  Mann-Whitney U={u:.1f}  p={p:.4f}  "
              f"difference of medians {np.median(c2s) - np.median(c2w):+.3f} m\n")

    print("=== Galileo error by the code resolving E5a ===")
    print(tabla.groupby("E2")[["E", "E/G"]].agg(["count", "median"]).round(3).to_string(), "\n")

    print("=== Radome ===")
    print(tabla.groupby("radomo")[["G", "E", "E/G"]].agg(["count", "median"]).round(3).to_string(), "\n")

    if args.salida:
        tabla.to_csv(f"{args.salida}.csv")
        print(f"saved to {args.salida}.csv")


if __name__ == "__main__":
    main()
