# TabPFN for Insurance Pricing </br><sub><sub> [Bruno Deprez, Wouter Verbeke, Tim Verdonck (2026)](https://arxiv.org/abs/TODO)</sub></sub>

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

The source code of the experimental evaluation accompanying the paper *TabPFN for Insurance Pricing*.
A preprint version of the work is available on arXiv at https://arxiv.org/abs/2605.22892.

This repository benchmarks the tabular foundation model **TabPFN** against the two standard baselines in insurance pricing — a generalised linear model (GLM) and gradient-boosted trees (XGBoost) — on claim frequency, claim severity, and binary telematics classification tasks. All three model families share a common cross-validation, evaluation, and SHAP-interpretability pipeline so that comparisons are like-for-like.

## Citing
Please cite our paper and/or code as follows:

```tex
@misc{deprez2026tabpfninsurance,
      title={Is TabPFN the Silver Bullet for Insurance Pricing?}, 
      author={Bruno Deprez and Wouter Verbeke and Tim Verdonck},
      year={2026},
      eprint={2605.22892},
      archivePrefix={arXiv},
      primaryClass={q-fin.RM},
      url={https://arxiv.org/abs/2605.22892}, 
}
```

## Data
The experiments are run on three datasets from the [`CASdatasets`](http://cas.uqam.ca/) R package: `freMTPL2` (French MTPL, frequency and severity), `beMTPL97` (Belgian MTPL, frequency and severity), and `fretelematic` (French telematics, binary classification).

This repository does not bundle any data. Datasets are loaded on demand from the R package via `rpy2` (see [src/data/loaders.py](src/data/loaders.py)) and cached as parquet files in `data/processed/` for fast subsequent access. Cleaning steps follow Wüthrich & Buser (2021) for `freMTPL2` and Henckaerts et al. (2021) for `beMTPL97`; `fretelematic` is used as-is. Installing the `CASdatasets` R package is part of the setup instructions below.

## Repository structure
The structure below lists folders and the scripts/notebooks containing code. Generated artefacts (parquet caches in `data/`, logs in `logs/`, result CSVs and plots in `res/`) are not shown.

```bash
|-config
    |-data.yaml
    |-features.yaml
    |-experiment_q1_severity.yaml
    |-experiment_q2_frequency.yaml
    |-experiment_q3_shap.yaml
    |-experiment_q4_fretelematic.yaml
|-data
|-logs
|-notebooks
    |-explore_categoricals.ipynb
    |-results_tables.ipynb
    |-test_telematics.ipynb
|-res
|-scripts
    |-run_q1_severity.py
    |-run_q2_frequency.py
    |-run_q3_shap.py
    |-run_q4_fretelematic.py
|-src
    |-data
        |-contracts.py
        |-cv.py
        |-loaders.py
        |-preprocessing.py
    |-methods
        |-glm_model.py
        |-tabpfn_model.py
        |-xgboost_model.py
    |-utils
        |-logging_setup.py
        |-metrics.py
        |-results.py
|-environment.yaml
|-environment.txt
```

## Installing
Setup proceeds in four steps. The order matters: `rpy2` requires `R` to be installed first, and `tabpfn` requires `torch` to be installed first. Full rationale and verification snippets are in [`environment.txt`](environment.txt).

**Step 1 — Create the conda environment (Python, R, rpy2, baseline packages).**

```bash
conda env create -f environment.yaml
conda activate tabpfn-insurance
```

**Step 2 — Install PyTorch (platform-specific).** Pick exactly one option.

```bash
# Apple Silicon (MPS, recommended for local dev)
pip install torch torchvision torchaudio

# Linux/Windows with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only (any platform, slowest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Step 3 — Install TabPFN (pip-only; must come after PyTorch).**

```bash
pip install tabpfn
```

**Step 4 — Install the `CASdatasets` R package.** `CASdatasets` is not on CRAN; it is fetched from the UQAM mirror. Its dependencies are installed via `conda-forge` (not via R's `install.packages`) so that pre-built binaries are used and no macOS SDK headers are required.

```bash
conda install -c conda-forge r-lattice r-survival r-xts r-zoo r-matrix
Rscript -e "options(timeout=600); install.packages('CASdatasets', repos='http://cas.uqam.ca/pub/', type='source')"
```

Optionally, register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name tabpfn-insurance --display-name "TabPFN Insurance"
```

Once installed, the four experiments can be run as standalone scripts:

```bash
python scripts/run_q1_severity.py
python scripts/run_q2_frequency.py
python scripts/run_q3_shap.py
python scripts/run_q4_fretelematic.py
```

Each script reads its YAML config from `config/`, writes per-fold metrics to `res/`, and emits a structured log to `logs/`. The [notebooks/results_tables.ipynb](notebooks/results_tables.ipynb) notebook compiles the CSVs in `res/` into the paper's tables and figures.

