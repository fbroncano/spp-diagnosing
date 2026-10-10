"""Exports the daily series of station CACE used in Figure 5.

A direct SQL query is not enough: `err_3d` is not stored in the database. It is
computed by joining the SPP position with the NGL reference series and rotating
the difference into the local ENU frame, which is what `dataset.build_dataset`
does. This script therefore goes through that single builder instead of reading
`daily_positions` by hand.

    python -m analysis.exportar_cace

Output: analysis/figure_data/F6_cace_serie.csv
"""
import os

import pandas as pd

from analysis.caracterizacion import cargar_todo

SALIDA = os.path.join(os.path.dirname(__file__), "figure_data", "F6_cace_serie.csv")


def main():
    df = cargar_todo(stations=["CACE"])
    sub = df[(df["method"] == "weighted_mad")
             & (df["constellation"].isin(["G", "E"]))]
    cols = ["date", "constellation", "err_3d", "d_e", "d_n", "d_u", "std_3d"]
    sub = sub[cols].sort_values(["date", "constellation"])
    sub.to_csv(os.path.normpath(SALIDA), index=False)
    print(f"{len(sub)} rows -> {os.path.normpath(SALIDA)}")
    print(sub.groupby("constellation")["err_3d"].describe()[
        ["count", "50%", "max"]].round(3))


if __name__ == "__main__":
    main()
