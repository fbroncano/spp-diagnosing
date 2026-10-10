"""Figures 3, 4 and 5 of the article.

The three figures are drawn from the CSV files in `analysis/figure_data/`,
without touching the PostGIS database, so they can be regenerated on any
machine with the repository.

    figures/Figure3.pdf  accuracy against reliability
    figures/Figure4.pdf  the degradation affects Galileo and not GPS
    figures/Figure5.pdf  CACE series around the antenna replacement

Figures 1 and 2 of the article (station map and pipeline diagram) were drawn by
hand and are not produced by this script.

Design decisions worth keeping:

  - Navy for GPS, orange for Galileo, dark outline with warm fill for GEC and
    for the legacy antennas.
  - Everything encoded with colour is also encoded with shape or a direct
    label: the article may be printed in greyscale.
  - Panel b of Figure 3 is a range chart, not a four-point scatter: the range
    from the median to the 99th percentile turns the tail into length, which
    is the magnitude the argument of the section rests on.
  - Panel a of Figure 4 is drawn without station labels, which were added
    afterwards. The position of each station is exported to
    `F4_panel_a_posiciones.csv` to place them.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(AQUI, ".."))
DATOS = os.path.join(AQUI, "figure_data")
# Figures are written with their submission names (Figure3.pdf...) to figures/.
SALIDA = os.path.join(REPO, "figures")
os.makedirs(SALIDA, exist_ok=True)

# The two complementary fill colours.
FRIO = "#c4cff3"
CALIDO = "#f3e8c4"

# Both fills have a contrast of only 1.6 and 1.2 against white, so on their own
# they would be the least visible element exactly where the reader should look.
# Emphasis is carried by the dark outline, not the fill, which also survives
# greyscale printing.
CONTORNO = "#52514e"

# Series. GEC takes the dark outline with warm fill, the highlight treatment.
GPS = "#1f4e79"
GALILEO = "#d1663c"
GEC = CONTORNO
BEIDOU = "#7a8899"
NEUTRO = "#c3c2b7"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
APAGADO = "#898781"

ORDEN_METODO = ["mean", "weighted", "median", "mad", "weighted_mad"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": APAGADO,
    "axes.labelcolor": TINTA,
    "axes.linewidth": 0.7,
    "xtick.color": TINTA_2,
    "ytick.color": TINTA_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
})


def limpiar(ax, dejar=("left", "bottom")):
    for lado, spine in ax.spines.items():
        spine.set_visible(lado in dejar)


def miles(x):
    """66071.657 -> '66,072'. The article uses a thousands separator."""
    return f"{round(x):,}"


# --------------------------------------------------------------------------
# Figure 3: accuracy against reliability
# --------------------------------------------------------------------------

def figura_3():
    crudo = pd.read_csv(os.path.join(DATOS, "F2_estimadores.csv"), header=[0, 1], index_col=0)
    crudo = crudo.drop(index="metodo", errors="ignore").astype(float)

    const = pd.read_csv(os.path.join(DATOS, "F4_constelaciones.csv"))
    const = const.set_index("constelacion")

    fig = plt.figure(figsize=(6.95, 5.75))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 0.92], hspace=0.62, wspace=0.30,
        left=0.145, right=0.965, top=0.90, bottom=0.085,
    )

    # ---- panel (a): per estimator ------------------------------------------
    columnas = [
        ("mediana", "Median 3D error (m)", False),
        ("p99", "99th percentile (m)", True),
        ("fallo_10m_pct", "Days with error > 10 m (%)", False),
    ]
    y = np.arange(len(ORDEN_METODO))[::-1]  # mean on top

    ejes_a = []
    for i, (metrica, etiqueta, log) in enumerate(columnas):
        ax = fig.add_subplot(gs[0, i])
        ejes_a.append(ax)
        # GPS drawn above GEC: where both are zero, the filled triangle used
        # to hide the blue dot.
        ax.scatter(crudo[(metrica, "G")].reindex(ORDEN_METODO), y,
                   marker="o", s=34, color=GPS, zorder=5, label="GPS")
        ax.scatter(crudo[(metrica, "GEC")].reindex(ORDEN_METODO), y,
                   marker="^", s=44, facecolors=CALIDO, edgecolors=GEC,
                   linewidths=1.2, zorder=3, label="Multi-constellation (GEC)")
        if log:
            ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(ORDEN_METODO if i == 0 else [])
        ax.set_ylim(-0.7, len(ORDEN_METODO) - 0.3)
        ax.set_xlabel(etiqueta, fontsize=8.5)
        ax.axhline(2.5, color=APAGADO, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.tick_params(length=2.5)
        limpiar(ax)

    ejes_a[0].text(-0.60, 0.965, "non-robust", transform=ejes_a[0].transAxes,
                   ha="left", va="center", fontsize=7.5, style="italic", color=APAGADO)
    ejes_a[0].text(-0.60, 0.60, "robust", transform=ejes_a[0].transAxes,
                   ha="left", va="center", fontsize=7.5, style="italic", color=APAGADO)

    ejes_a[1].annotate(
        f"daily maximum\n{miles(66071.657)} m (GPS)\n{miles(492455.639)} m (GEC)",
        xy=(781.245, 3), xycoords="data", xytext=(0.24, 0.26), textcoords="axes fraction",
        fontsize=7, color=TINTA_2, ha="left", va="center", linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=APAGADO, lw=0.7,
                        connectionstyle="arc3,rad=0.25"),
    )

    fig.text(0.018, 0.975, "(a)", fontsize=10, fontweight="bold", va="top")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", color=GPS, markersize=6, label="GPS"),
            Line2D([], [], marker="^", ls="none", markerfacecolor=CALIDO,
                   markeredgecolor=GEC, markeredgewidth=1.2, markersize=7.5,
                   label="Multi-constellation (GEC)"),
        ],
        loc="upper center", bbox_to_anchor=(0.57, 1.005), ncol=2,
        frameon=False, fontsize=8.5, handletextpad=0.4, columnspacing=1.8,
    )

    # ---- panel (b): ranges per constellation ------------------------------
    ax = fig.add_subplot(gs[1, :])
    # The two numeric margin columns live outside the data area, so the axis is
    # narrowed to make room for them without overlapping the "max" labels.
    caja = ax.get_position()
    ax.set_position([caja.x0, caja.y0, caja.width * 0.775, caja.height])
    filas = [
        ("GPS", "G", GPS, FRIO),
        ("Galileo", "E", GALILEO, CALIDO),
        ("Multi-constellation", "GEC", GEC, CALIDO),
        ("BeiDou", "C", BEIDOU, FRIO),
    ]
    yy = np.arange(len(filas))[::-1]

    for (nombre, clave, color, pastel), yi in zip(filas, yy):
        r = const.loc[clave]
        ax.plot([r["mediana_3d"], r["p99_3d"]], [yi, yi], color=color, lw=3.2,
                solid_capstyle="round", zorder=2)
        ax.plot([r["p99_3d"], r["max_3d"]], [yi, yi], color=color, lw=0.8,
                ls=(0, (1.5, 1.5)), zorder=2)
        ax.plot([r["max_3d"]], [yi], marker="|", markersize=6, color=color,
                markeredgewidth=1.1, zorder=3)
        ax.plot([r["mediana_3d"]], [yi], marker="o", markersize=7, color=color, zorder=4)
        ax.plot([r["p99_3d"]], [yi], marker="o", markersize=7, markerfacecolor=pastel,
                markeredgecolor=color, markeredgewidth=1.6, zorder=4)
        # The maximum goes below its mark rather than to the right, where the
        # BeiDou one (63.4) collided with the failed-days column.
        ax.text(r["max_3d"], yi - 0.24, f"max {r['max_3d']:.1f}", fontsize=7,
                color=APAGADO, va="top", ha="center")

    ax.set_xscale("log")
    ax.set_xlim(0.62, 92)
    ax.set_ylim(-1.0, len(filas) - 0.3)
    ax.set_yticks(yy)
    ax.set_yticklabels([n for n, _, _, _ in filas])
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_xlabel("Median 3D error  →  99th percentile (m), logarithmic", fontsize=8.5)
    ax.tick_params(length=2.5)
    limpiar(ax)

    # Two numeric margin columns give the third and fourth figures without extra
    # axes. The satellite column supports the "2.6 times more" statement.
    x_fallo, x_sat = 1.105, 1.265
    ax.text(x_fallo, 1.02, "days\n> 10 m", transform=ax.transAxes, fontsize=7.5,
            color=TINTA_2, ha="center", va="bottom", linespacing=1.3)
    ax.text(x_sat, 1.02, "sat per\nepoch", transform=ax.transAxes, fontsize=7.5,
            color=TINTA_2, ha="center", va="bottom", linespacing=1.3)
    for (nombre, clave, color, pastel), yi in zip(filas, yy):
        r = const.loc[clave]
        frac = (yi - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        resaltar = r["fallo_10m_%"] == 0
        ax.text(x_fallo, frac, f"{r['fallo_10m_%']:.2f} %", transform=ax.transAxes,
                fontsize=8, ha="center", va="center",
                color=TINTA if resaltar else TINTA_2,
                fontweight="bold" if resaltar else "normal")
        ax.text(x_sat, frac, f"{r['n_sat']:.1f}", transform=ax.transAxes,
                fontsize=8, ha="center", va="center", color=TINTA_2)

    ax.annotate(
        "2.6× more satellites:\n+89 % median error,\nno day above 10 m",
        xy=(const.loc["GEC", "mediana_3d"], 1.0), xycoords="data",
        xytext=(0.66, 0.42), textcoords="data",
        fontsize=7.5, color=TINTA_2, ha="left", va="center", linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=APAGADO, lw=0.7,
                        connectionstyle="arc3,rad=-0.3"),
    )

    fig.text(0.018, 0.455, "(b)", fontsize=10, fontweight="bold", va="top")

    destino = os.path.join(SALIDA, "Figure3.pdf")
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------
# Figure 4: the degradation affects Galileo and not GPS
# --------------------------------------------------------------------------

def figura_4():
    est = pd.read_csv(os.path.join(DATOS, "F3_N1_por_estacion.csv"))
    ant = pd.read_csv(os.path.join(DATOS, "F5_por_antena.csv"))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(8.25, 3.70), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.30)
    )
    fig.subplots_adjust(left=0.085, right=0.975, top=0.92, bottom=0.175)

    # ---- panel (a): scatter, without labels ------------------------------
    estilo = {
        "heredada": dict(marker="o", s=52, facecolors=CALIDO, edgecolors=CONTORNO,
                         linewidths=1.3, zorder=5),
        "sustituida": dict(marker="D", s=48, facecolors=FRIO, edgecolors=CONTORNO,
                           linewidths=1.3, zorder=4),
        "moderna": dict(marker="o", s=34, facecolors="none", edgecolors=APAGADO,
                        linewidths=1.0, zorder=3),
    }
    for estado, kw in estilo.items():
        sub = est[est["estado_antena"] == estado]
        ax_a.scatter(sub["G"], sub["E"], **kw)

    lim = np.array([0.55, 1.45])
    ax_a.plot(lim, lim, ls=(0, (5, 4)), color=NEUTRO, lw=0.9, zorder=1)
    ax_a.set_xlim(*lim)
    ax_a.set_ylim(0.3, 12.4)
    ax_a.set_xlabel("Median 3D error, GPS (m)", fontsize=8.5)
    ax_a.set_ylabel("Median 3D error, Galileo (m)", fontsize=8.5)
    ax_a.tick_params(length=2.5)
    limpiar(ax_a)

    # ---- panel (b): per antenna model -------------------------------------
    ant = ant.sort_values("E/G", ascending=True).reset_index(drop=True)
    yb = np.arange(len(ant))
    destacada = [v >= 2.0 for v in ant["E/G"]]
    colores = [CALIDO if d else NEUTRO for d in destacada]
    bordes = [CONTORNO if d else APAGADO for d in destacada]
    grosores = [1.0 if d else 0.5 for d in destacada]
    ax_b.barh(yb, ant["E/G"] - 1.0, left=1.0, color=colores, edgecolor=bordes,
              linewidth=grosores, height=0.62, zorder=2)

    etiquetas = [
        f"{m} *" if m == "LEIAR25" else m for m in ant["antenna_type"]
    ]
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(etiquetas)
    for yi, (v, n) in enumerate(zip(ant["E/G"], ant["n_est"])):
        ax_b.text(v + 0.22, yi, f"{v:.2f}  ({n})", va="center", ha="left",
                  fontsize=8, color=TINTA_2)

    ax_b.axvline(1.0, color=APAGADO, lw=0.8, zorder=3)
    ax_b.set_xlim(1.0, 13.2)
    ax_b.set_ylim(-0.8, len(ant) + 0.25)
    ax_b.set_xticks([2, 4, 6, 8, 10, 12])
    ax_b.set_xlabel("E/G ratio of the median 3D error", fontsize=8.5)
    ax_b.text(1.12, len(ant) - 0.25, "no degradation", fontsize=7.5, color=APAGADO,
              ha="left", va="center")
    ax_b.tick_params(length=2.5)
    limpiar(ax_b)

    fig.text(0.012, 0.975, "(a)", fontsize=10, fontweight="bold", va="top")
    fig.text(0.512, 0.975, "(b)", fontsize=10, fontweight="bold", va="top")
    fig.text(
        0.975, 0.012,
        "*  averages two sites that differ by a factor of three (LEON 3.36, SALA 1.19)",
        fontsize=7.5, color=APAGADO, ha="right", va="bottom",
    )

    destino = os.path.join(SALIDA, "Figure4.pdf")
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)

    # Position of each station in panel a, for labelling.
    fig.canvas.draw()
    filas = []
    ancho, alto = fig.canvas.get_width_height()
    for _, r in est.iterrows():
        px, py = ax_a.transData.transform((r["G"], r["E"]))
        fx, fy = fig.transFigure.inverted().transform((px, py))
        filas.append({
            "station_id": r["station_id"],
            "antenna": r["antena"],
            "antenna_status": {"heredada": "legacy", "sustituida": "replaced",
                               "moderna": "modern"}[r["estado_antena"]],
            "gps_3d_m": r["G"],
            "galileo_3d_m": r["E"],
            "fig_frac_x": round(fx, 5),
            "fig_frac_y_from_bottom": round(fy, 5),
            "px_x_at_300dpi": round(px / fig.dpi * 300, 1),
            "px_y_from_top_at_300dpi": round((alto - py) / fig.dpi * 300, 1),
        })
    pos = pd.DataFrame(filas).sort_values("galileo_3d_m", ascending=False)
    csv_pos = os.path.join(DATOS, "F4_panel_a_posiciones.csv")
    pos.to_csv(csv_pos, index=False)
    plt.close(fig)
    return destino, csv_pos


# --------------------------------------------------------------------------
# Figure 5: CACE series
# --------------------------------------------------------------------------

def figura_5():
    s = pd.read_csv(os.path.join(DATOS, "F6_cace_serie.csv"), parse_dates=["date"])

    cambio = pd.Timestamp("2025-03-19")
    segunda = pd.Timestamp("2026-01-01")

    fig, ax = plt.subplots(figsize=(7.30, 3.25))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.135)

    # The two regimes after the replacement, shaded: they show as a region what
    # the two vertical lines mark as an instant.
    ax.axvspan(cambio, segunda, color=CALIDO, alpha=0.55, lw=0, zorder=0)
    ax.axvspan(segunda, s["date"].max(), color=FRIO, alpha=0.55, lw=0, zorder=0)

    series = [("G", "GPS", GPS, "-"), ("E", "Galileo", GALILEO, (0, (5, 2)))]
    for clave, nombre, color, trazo in series:
        d = s[s["constellation"] == clave].sort_values("date")
        ax.scatter(d["date"], d["err_3d"], s=2.2, color=color, alpha=0.20,
                   linewidths=0, zorder=2)
        movil = (d.set_index("date")["err_3d"]
                   .rolling("15D", center=True, min_periods=5).median())
        ax.plot(movil.index, movil.values, color=color, lw=1.4, ls=trazo,
                zorder=4, label=nombre)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 10]))
    ax.set_yscale("log")
    ax.set_ylabel("Daily 3D error (m)", fontsize=8.5)
    ax.set_ylim(0.03, 30)
    ax.set_xlim(s["date"].min() - pd.Timedelta(days=15),
                s["date"].max() + pd.Timedelta(days=15))
    ax.tick_params(length=2.5)
    limpiar(ax)

    ax.axvline(cambio, color=TINTA, lw=0.9, ls=(0, (6, 3)), zorder=5)
    ax.axvline(segunda, color=TINTA, lw=0.9, ls=(0, (1.5, 1.8)), zorder=5)
    ax.text(cambio - pd.Timedelta(days=22), 21,
            "19 Mar 2025\nantenna and receiver\nreplaced (site log)",
            fontsize=7.5, color=TINTA_2, ha="right", va="top", linespacing=1.35)
    ax.text(segunda + pd.Timedelta(days=22), 21,
            "Jan 2026\nno change recorded\nin the site logs",
            fontsize=7.5, color=TINTA_2, ha="left", va="top", linespacing=1.35)

    tramos = [
        (s["date"].min(), cambio, 3.94, 0.738, 2.904),
        (cambio, segunda, 0.90, 0.205, 0.185),
        (segunda, s["date"].max(), 1.05, 0.873, 0.916),
    ]
    for ini, fin, eg, g, e in tramos:
        centro = ini + (fin - ini) / 2
        ax.text(centro, 0.047,
                f"E/G = {eg:.2f}\nGPS {g:.3f} m\nGalileo {e:.3f} m",
                fontsize=7.5, color=TINTA_2, ha="center", va="bottom",
                linespacing=1.35)

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=2, frameon=False,
              fontsize=8.5, handletextpad=0.6, columnspacing=1.6)

    destino = os.path.join(SALIDA, "Figure5.pdf")
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return destino


def comprobar_medianas():
    """Recomputes from the series the medians written on Figure 5."""
    s = pd.read_csv(os.path.join(DATOS, "F6_cace_serie.csv"), parse_dates=["date"])
    cortes = [(None, "2025-03-19"), ("2025-03-19", "2026-01-01"), ("2026-01-01", None)]
    print("\ncheck of the three CACE periods")
    for ini, fin in cortes:
        d = s.copy()
        if ini:
            d = d[d["date"] >= ini]
        if fin:
            d = d[d["date"] < fin]
        g = d[d["constellation"] == "G"]["err_3d"].median()
        e = d[d["constellation"] == "E"]["err_3d"].median()
        n = len(d[d["constellation"] == "G"])
        print(f"  {str(ini or 'start'):12s} to {str(fin or 'end'):12s} "
              f"n={n:4d}  GPS {g:.3f}  Galileo {e:.3f}  E/G {e/g:.2f}")


if __name__ == "__main__":
    comprobar_medianas()
    print("\n" + figura_3())
    pdf, csv_pos = figura_4()
    print(pdf)
    print(csv_pos)
    print(figura_5())
