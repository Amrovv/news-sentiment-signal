# stock-predictor

ML model to predict stock prices

## Project Organization

├── .env               <- Secret/config variables (git-ignored)
├── .gitignore         <- Files git should never track
├── Makefile           <- Shortcut commands: make data, make test, make lint
├── README.md          <- Top-level project readme
├── pyproject.toml     <- Project metadata, dependencies, tool config
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
│   ├── config.py      <- Central paths, logging, env loading
│   ├── dataset.py     <- Download / generate data       (raw → processed)
│   ├── features.py    <- Build model features           (processed → features)
│   ├── plots.py       <- Create visualizations          (processed → figures)
│   └── modeling
│       ├── __init__.py
│       ├── train.py   <- Train models                   (features → model.pkl)
│       └── predict.py <- Run inference                  (model.pkl → predictions)
│
└── tests              <- Automated tests (pytest)

--------

