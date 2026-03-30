# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Orion

Orion (`orion-ml` on PyPI) is an unsupervised time series anomaly detection library from MIT's Data to AI Lab. It provides modular ML pipelines (built on the MLBlocks framework) that identify rare patterns in time series data. The primary dataset is 75 labeled NASA signals.

## Commands

### Installation

```bash
make install-develop          # Editable install with all dev dependencies (standard for development)
make install-pretrained-develop  # Editable install + pretrained model deps (TimesFM, Chronos2, etc.)
```

### Testing

```bash
make test-unit                # Run unit tests with coverage: pytest ./tests/unit --cov=orion
pytest tests/unit/test_core.py  # Run a single test file
pytest tests/unit/test_core.py::TestOrion::test_fit  # Run a single test
make test-readme              # Test README code examples
make test-pretrained          # Test pretrained model pipelines
make test                     # Unit + readme + tutorials
```

### Linting

```bash
make lint                     # flake8 + isort check (max line length: 99)
make fix-lint                 # autoflake + autopep8 + isort auto-fix
```

### Other

```bash
make coverage                 # Generate HTML coverage report
make docs                     # Build Sphinx documentation
orion evaluate -p aer -s S-1 -m f1  # CLI: evaluate a pipeline on a signal
```

## Architecture

### Pipeline System

Orion uses **MLBlocks pipelines** — JSON-defined DAGs of primitives executed sequentially. Each pipeline JSON specifies:
- `primitives`: ordered list of primitive class names
- `init_params`: per-primitive hyperparameters
- `input_names` / `output_names`: data flow mappings between primitives

Pipelines live in `orion/pipelines/`:
- `verified/` — 9 production-ready pipelines (AER, TadGAN, LSTM, VAE, etc.)
- `pretrained/` — pipelines using foundation models (TimesFM, Chronos2)
- `sandbox/` — experimental pipelines

The best-performing pipeline is **AER** (Auto-Encoder with Regression), which wins on all 12 benchmark datasets vs. the ARIMA baseline.

### Core Classes

**`orion/core.py` — `Orion` class**: Main entry point. Wraps an MLBlocks `MLPipeline`. Key methods: `fit(data)`, `detect(data)`, `score(true_anomalies, detected_anomalies)`, `save(path)` / `load(path)`. Default pipeline is `lstm_dynamic_threshold`.

**`orion/functional.py`**: Simplified functional API — `fit_pipeline()`, `detect_anomalies()`, `evaluate_pipeline()`.

**`orion/benchmark.py`**: Evaluates multiple pipelines across multiple datasets. Supports multiprocessing/dask parallelism and result caching.

**`orion/data.py`**: Loads NASA signals from S3 (cached locally in `orion/data/`). Also handles custom CSVs via `load_csv()` and `format_csv()`.

### Primitives (`orion/primitives/`)

Custom ML components that are composed into pipelines. Each primitive is a Python class with a JSON spec in `orion/primitives/jsons/`. Categories:

- **Models**: `aer.py`, `tadgan.py`, `vae.py`, `anomaly_transformer.py`, `chronos2.py`, `timesfm.py`, `cisco.py`
- **Preprocessing**: `timeseries_preprocessing.py` (windowing, scaling, imputation)
- **Scoring/Detection**: `timeseries_errors.py` (DTW/Euclidean reconstruction error), `timeseries_anomalies.py` (thresholding → anomaly intervals)
- **Adapters**: `adapters/` (NCP neural circuit polynomials)

### Evaluation (`orion/evaluation/`)

Two anomaly evaluation paradigms:
- **Point** (`point.py`): timestamp-level detection metrics (accuracy, precision, recall, F1)
- **Contextual** (`contextual.py`): interval-based metrics using weighted/overlap segment approaches

### Data Format

Time series data is a pandas DataFrame with columns:
- `timestamp` — Unix timestamps (int64)
- `value` — float signal values

Anomalies are DataFrames with `start` and `end` Unix timestamp columns.
