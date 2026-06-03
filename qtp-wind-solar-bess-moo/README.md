# Spatial Multi-Objective Optimization for Wind-Solar-BESS Siting on the QTP

This repository contains code supporting the manuscript:

> A spatial multi-objective optimization and solution-fusion framework for wind and solar energy siting on the Qinghai-Tibet Plateau

The code implements a binary spatial multi-objective optimization framework for
regional renewable-energy siting. It compares three environmental selection
strategies under a shared binary encoding and DE-inspired offspring generation
procedure:

- NSGA-II
- MOEA/D
- SPEA2-style strength-density environmental selection

The three objectives are formulated in minimization form:

1. negative annual energy production,
2. negative regional net economic return,
3. grid-connection distance.

## Repository Structure

```text
.
├── src/qtp_moo/               # Core optimization functions
├── scripts/run_experiment.py  # Command-line experiment runner
├── data/README.md             # Required input variables and data notes
├── notebooks/                 # Cleaned provenance notebook
├── results/README.md          # Placeholder for generated outputs
├── CODE_AVAILABILITY.md       # Text and fields for publication
├── CITATION.cff               # Citation metadata template
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On Linux/macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Input Data

Place processed NetCDF files such as `MPDEV2018.nc` in the `data/` directory.
The expected variables are listed in `data/README.md`.

The GitHub repository does not include large processed raster files or derived
result files. These should be regenerated from the manuscript data workflow or
made available through a research data repository if journal policy requires
full reproducibility from processed inputs.

## Example Run

```bash
python scripts/run_experiment.py --input data/MPDEV2018.nc --output results/MPDEV2018 --runs 10
```

The script writes per-run D-indicator CSV files and compressed population
history files under `results/`.

## Method Note

The offspring operator is adapted to binary decision vectors and should be
described as a DE-inspired binary recombination operator rather than a standard
continuous-space differential evolution operator.

## License

This code is released under the MIT License. Update `CITATION.cff` and
`CODE_AVAILABILITY.md` with the final repository URL, authors, and archived
release DOI before submission.
