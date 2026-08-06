# stock-predictor

ML model to predict stock prices

## Project Organization
```
├── .env               <- Secret/config variables (git-ignored)
├── .gitignore         <- Files git should never track
├── Makefile           <- Shortcut commands: make data, make test, make lint
├── README.md          <- Top-level project readme
├── pyproject.toml     <- Project metadata, dependencies, tool config
├── FINDINGS.md        <- Running results log
│
├── app                <- Streamlit demo
│
├── data
│   ├── external       <- Data from third-party sources
│   ├── interim        <- Intermediate, partially transformed data
│   ├── processed      <- Final, canonical datasets for modeling
│   └── raw            <- Original, immutable data dump
│
├── docs               <- Project documentation (mkdocs-style)
├── models             <- Trained/serialized models + predictions
├── notebooks          <- Jupyter notebooks for exploration
├── references         <- Data dictionaries, manuals, explanatory material
├── reports            <- Generated analysis (HTML/PDF/...)
│   └── figures        <- Generated graphics and figures
│
├── stock_predictor    <- Source code package
│   ├── __init__.py    <- Marks the folder as a Python package
│   ├── config.py      <- Central paths, logging, env loading, constants
│   ├── text           <- Person A layer (text & deep learning)
│   ├── market         <- Person B layer (market & modeling)
│   ├── dataset.py     <- Download / generate data       (raw → processed)
│   ├── features.py    <- Build model features           (processed → features)
│   ├── plots.py       <- Create visualizations          (processed → figures)
│   └── modeling
│       ├── __init__.py
│       ├── train.py   <- Train models                   (features → model.pkl)
│       └── predict.py <- Run inference                  (model.pkl → predictions)
│
└── tests              <- Automated tests (pytest)
```
--------

## Setup

```bash
uv sync
```

`uv sync` installs everything in `pyproject.toml`. A few things it does **not** cover:

**1. spaCy language model** — needed by the entity filter; not a pip dependency:
```bash
uv run python -m spacy download en_core_web_sm
```

**2. FinBERT weights** — `ProsusAI/finbert` auto-downloads from HuggingFace to your
local cache on first use. To pre-fetch (optional, avoids a stall on first run):
```bash
uv run python -c "from transformers import AutoModelForSequenceClassification as M; M.from_pretrained('ProsusAI/finbert')"
```

**3. Finnhub API key** — create `.env` (git-ignored) with your free key from finnhub.io:
```
FINNHUB_API_KEY=your_key_here
```

