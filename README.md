# stock-predictor

Predicting TSLA price movement from financial news sentiment.

The text layer is built and is where the work has gone: it turns 2,124 scraped articles into one
feature row per article, scoring each sentence for sentiment *toward the target company* rather than
sentiment in general, and verifying with a local LLM that each coreference-resolved sentence really
is about that company. The market and modeling layers are scaffolding. **No price model has been
trained yet**, so every accuracy figure in this repository is a text metric, never a trading result.

The pipeline is `stock_predictor/text/`, driven end to end by `run_pipeline.py`. The reasoning
behind every choice in it lives in `notebooks/text/`, in pipeline order.

## Setup

```bash
uv sync
```

Four things `uv sync` does not cover:

**1. spaCy language model**, needed by the entity filter:

```bash
uv run python -m spacy download en_core_web_sm
```

**2. Transformer weights.** FinBERT (`ProsusAI/finbert`) and the ABSA model
(`yangheng/deberta-v3-base-absa-v1.1`) download from HuggingFace on first use. The coreference model
(`biu-nlp/f-coref`) does the same. To pre-fetch and avoid a stall mid-run:

```bash
uv run python -c "from transformers import AutoModelForSequenceClassification as M; M.from_pretrained('ProsusAI/finbert'); M.from_pretrained('yangheng/deberta-v3-base-absa-v1.1')"
```

**3. The referent-verification judge**, a 4-bit GGUF of Qwen2.5-7B-Instruct, about 4.7 GB. It is not
fetched automatically and `models/` is gitignored. Place it at the path `config.JUDGE_MODEL_PATH`
expects:

```
models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

Without it `coref_judge.is_available()` returns False and the judge stage is skipped, which leaves
the corpus unverified rather than failing loudly.

**4. Finnhub API key**, for re-pulling the article feed. Create `.env` (gitignored):

```
FINNHUB_API_KEY=your_key_here
```

## Running the text pipeline

```bash
uv run python -m stock_predictor.text.run_pipeline
```

It reads the cleaned article table at `config.PROCESSED_ARTICLES_PATH` and runs five phases: split
and tag, score with both models, judge the coreference-resolved sentences, aggregate to articles,
and write the deliverable.

**Budget about five hours from cold**, almost all of it the judge at roughly 5 seconds per sentence
on CPU. With the caches in `data/interim/` warm it is closer to 25 minutes. The run is resumable:
phases skip if their output exists, and the judge flushes verdicts every 100 rows, so an interrupted
run picks up where it stopped rather than restarting.

The output is:

```
data/processed/pipeline_run/article_features.parquet    1,976 articles x 23 columns
data/processed/pipeline_run/article_features.md         its data dictionary
```

One row per article, judge-gated, every score a fusion of both scorers. Read the `.md` before
modelling on it: several columns are deliberately NaN rather than 0, and one is deliberately the
reverse.

## Where things are documented

- `data/README.md` — what each data directory holds, what is tracked, what rebuilding costs
- `references/README.md` — the label sets, which is authoritative, and how `sample_id` decodes
- `notebooks/text/` — the decisions and their evidence, in pipeline order. `2.0` builds the sentence
  table, `2.1` resolves who each sentence is about, `2.2` chooses the score, `2.3` builds the article
  table. The 2.x notebooks are a record with stored output and are not re-runnable.

## Project organisation

```
├── .env               <- Secret/config variables (git-ignored)
├── Makefile           <- Shortcut commands: make lint, make test, make format
├── pyproject.toml     <- Project metadata, dependencies, tool config
│
├── app                <- Streamlit demo (not built)
│
├── data               <- See data/README.md
│   ├── raw            <- The Finnhub pull. The only true source.
│   ├── interim        <- Caches and pipeline phase outputs (git-ignored)
│   ├── processed      <- pipeline_run/ holds the deliverable
│   ├── eval           <- Labelled referent ground truth
│   └── external       <- Third-party data (empty)
│
├── docs               <- Project documentation (empty)
├── models             <- Model weights, incl. the judge GGUF (git-ignored)
├── notebooks          <- text/ is Person A, market/ is Person B
├── references         <- Label sets and the labelling protocol
├── reports            <- Generated analysis
│   └── figures        <- Generated graphics
│
├── stock_predictor    <- Source code package
│   ├── config.py      <- Central paths, constants, logging
│   ├── text           <- Person A layer (text & deep learning)
│   ├── market         <- Person B layer (market data)
│   ├── modeling       <- Train / predict (stubs, not implemented)
│   ├── dataset.py     <- Download / generate data (stub)
│   ├── features.py    <- Build model features (stub)
│   └── plots.py       <- Create visualizations (stub)
│
└── tests              <- Automated tests (pytest)
```

Two people work in this repo. **Do not edit `stock_predictor/market/`** — it belongs to Person B.
