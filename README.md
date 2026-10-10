# SPP pipeline audit over the Spanish EPN stations

Code accompanying the article:

> F. Broncano, P. G. Rodríguez, M. Botejara-Antúnez, A. Caro and A. Cuartero,
> *Diagnosing a GNSS data processing pipeline through instrument metadata over
> the EPN network in Spain* (submitted).

It contains the standard point positioning (SPP) pipeline that was audited and
the analysis that produces the tables and figures of the article. Positions are
computed for each station from its own observation and navigation RINEX files,
without precise products, for GPS, Galileo, BeiDou and their combination, with
five daily estimators. They are compared against the daily PPP series of the
Nevada Geodetic Laboratory (NGL).

## Repository structure

```
├── processor/                 SPP pipeline
│   ├── main.py                entry point: processes station-days and writes to PostGIS
│   ├── manage_files.py        download and reading of RINEX files (IGN ERGNSS server)
│   ├── pre_processor.py       observable selection, ionosphere-free combination, troposphere, Sagnac
│   ├── sv_processor.py        satellite positions from broadcast ephemerides
│   ├── glonass.py             GLONASS orbit propagation (not used in the article)
│   ├── hatchfilter.py         carrier smoothing of code (Hatch filter)
│   ├── gnss_processor.py      epoch-wise least-squares solution
│   ├── post_processor.py      daily estimators, DOP and quality statistics
│   ├── utils.py               time, file-name and coordinate utilities
│   ├── CONSTANTS.py           configuration (decimation, Hatch window, frequency pairs...)
│   ├── dao.py, models.py      data-access layer over PostGIS
│   ├── load_stations.py       loads the station catalogue (estaciones_itrf2014.csv)
│   └── load_sitelogs.py       loads antenna and receiver history from EPN site logs
├── analysis/                  analysis of the article
│   ├── reference.py           downloads the NGL reference series into data/ngl/
│   ├── dataset.py             joins SPP solutions with the reference, ENU error
│   ├── geodesy.py             ECEF/geodetic/ENU conversions
│   ├── caracterizacion.py     the 20 constellation-estimator combinations, per station and per antenna
│   ├── receptor.py            antenna vs receiver, bootstrap intervals, permutation tests
│   ├── observables.py         observable resolved per band from the RINEX headers
│   ├── exportar_cace.py       daily series of CACE
│   ├── generate_figures.py    Figures 3, 4 and 5
│   └── figure_data/           CSV files behind Figures 3, 4 and 5
├── patches/
│   └── georinex-nav3.patch    fix for the BeiDou ephemeris defect in georinex 1.16.2
├── postgis.cfg.example        database connection template
└── requirements.txt
```

## Requirements

Python 3.12 and the packages in `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**`georinex` must be patched.** In version 1.16.2, the RINEX 3 navigation reader
discards an entire ephemeris when an empty field is padded with spaces, which
is how BeiDou writes it: without the patch no BeiDou ephemeris is read, and the
only trace is an INFO-level message (`malformed line for Cxx`). Apply it to the
installed package:

```bash
SITE="$(python -c 'import georinex, os; print(os.path.dirname(os.path.dirname(georinex.__file__)))')"
patch -p2 --directory="$SITE" < patches/georinex-nav3.patch
```

(or clone georinex 1.16.2, apply the patch with `patch -p1` and install it with
`pip install .`).

## Database

Solutions, quality statistics and station metadata are stored in PostgreSQL
with PostGIS. This repository contains only the data-access layer
(`processor/dao.py`). The schema, the SQL functions used by the DAO and the
containerised deployment are published in
[GeoGNSS-PS](https://github.com/fbroncano/GeoGNSS-PS).

Copy the connection template and fill in your values; `postgis.cfg` is ignored
by git:

```bash
cp postgis.cfg.example postgis.cfg
```

The file is looked up from `processor/` upwards, and any `POSTGRES_*`
environment variable overrides it.

## Reproducing the article

```bash
# 1. Station metadata
cd processor
python load_stations.py                       # station catalogue
python load_sitelogs.py --dir <site_logs>     # EPN site logs (https://epncb.oma.be/ftp/station/log/)

# 2. SPP processing (downloads RINEX files from the IGN server, 300 s decimation)
python main.py --start 2023-10-01 --end 2026-06-27 --workers 12
cd ..

# 3. Reference series
python -m analysis.reference

# 4. Analysis
python -m analysis.caracterizacion --salida results/caracterizacion
python -m analysis.caracterizacion --comun --salida results/caracterizacion_comun
python -m analysis.receptor --salida results/receptor
python -m analysis.observables --cabeceras <headers_dir> --salida results/observables
python -m analysis.exportar_cace

# 5. Figures 3-5, from analysis/figure_data/ (written to figures/)
python -m analysis.generate_figures
```

`main.py` processes all 27 stations over the study period by default. Each run
is registered in the `runs` table, and every position and statistic carries its
`run_id`. A station-day whose files are missing on the server is logged as
`[WORKER ERROR]` and does not stop the run.

`observables.py` expects one text file per station (`<STATION>.txt`) containing
the header of one of its RINEX observation files.

| Article element | Script |
|---|---|
| Table 2 (20 combinations) | `caracterizacion.py` |
| Paired comparison over 23 stations | `caracterizacion.py --comun` |
| Table 3 (per antenna model) | `caracterizacion.py` (`*_por_antena.csv`) |
| Permutation tests, bootstrap intervals, receiver checks (S4) | `receptor.py` |
| Observable codes and the C2S/C2W check | `observables.py` |
| Figures 3, 4 and 5 | `generate_figures.py` |

Figures 1 and 2 (station map and pipeline diagram) were drawn by hand and are
not generated by this code. The CSV files in `analysis/figure_data/` are
derived from the outputs of `caracterizacion.py` (per-estimator, per-station
and per-antenna tables) and `exportar_cace.py` (CACE series).

Some identifiers, command-line options and output column names keep their
original Spanish names (for example `--salida` for the output prefix, or
`mediana_3d` for the median 3D error) so that the outputs match those used to
prepare the article.

## Known defect

For each band, the pipeline selects the first observation code in the RINEX
header that matches the band prefix, regardless of the tracking attribute. At
stations that declare `C2S` before `C2W`, the second GPS frequency is
therefore resolved to the civil code rather than to the P(Y) code to which the
broadcast clock refers. The article discusses its effect (Section 5.1). The
behaviour is kept as published so that the results can be reproduced.

## Data sources

- GNSS observation and navigation files: Instituto Geográfico Nacional, ERGNSS
  network (`https://datos-geodesia.ign.es/ERGNSS/`).
- Station site logs: EPN Central Bureau (`https://epncb.oma.be/`).
- Reference series: Nevada Geodetic Laboratory, IGS20 `tenv3` series
  (`https://geodesy.unr.edu/`). Downloaded by `analysis/reference.py`, not
  redistributed.

## Licence and citation

MIT licence (see `LICENSE`). If you use this code, please cite the article and
the archived release (`CITATION.cff`).
