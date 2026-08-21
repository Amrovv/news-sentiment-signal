import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Project constants — single source of truth (no magic constants elsewhere)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Tickers
PRIMARY_TICKER = "TSLA"
TRANSFER_TICKERS = ["NVDA"]  # cross-firm test
CONTROL_TICKER = "KO"  # low-news control
MARKET_INDEX = "SPY"  # abnormal-return benchmark

# --- Company registry: one symmetric roster, any entry can be the target -----
#
# Keyed by ticker for public companies, canonical name for private ones. Selected
# at runtime by `ticker`, which is what keeps the pipeline ticker-agnostic.
#
# Tiers, consumed by entity_filter:
#   "names"     unambiguous aliases. A match sets mentions_target for the target.
#   "person"    associated people. Ambiguous, since "Musk" may mean SpaceX, so a
#               match sets mentions_ceo only and NEVER mentions_target.
#   "products"  product and technology names. Purely subtractive: never sets a
#               flag, never becomes an antecedent. Stops a product surface being
#               substituted with the company name, so "whether FSD would stop for
#               a school bus" is not rewritten into "would stop for Tesla". The
#               union across every entry applies on every run, since a product is
#               never the company whoever owns it.
#
# "person" and "products" are optional per entry.
COMPANIES = {
    "TSLA": {
        "names": ["Tesla", "TSLA", "Tesla Inc", "Tesla, Inc.", "$TSLA"],
        "person": ["Musk", "Elon Musk", "Elon"],
        "products": [
            "Model S",
            "Model 3",
            "Model X",
            "Model Y",
            "Cybertruck",
            "Cybercab",
            "Semi",
            "Roadster",
            "Powerwall",
            "Megapack",
            "Megafactory",
            "Optimus",
            "FSD",
            "Full Self-Driving",
            "Autopilot",
            "Robotaxi",
            "Supercharger",
            "Dojo",
        ],
    },
    "NVDA": {
        "names": ["Nvidia", "NVDA", "Nvidia Corp", "$NVDA"],
        "person": ["Jensen Huang", "Huang"],
        "products": ["Blackwell", "Hopper", "CUDA", "GeForce"],
    },
    "KO": {
        "names": ["Coca-Cola", "Coca Cola", "Coke", "KO", "$KO"],
        "person": [],
    },
    "BYD": {"names": ["BYD"]},
    "F": {"names": ["Ford", "Ford Motor"]},
    "GM": {"names": ["General Motors", "GM"]},
    "RIVN": {
        "names": ["Rivian"],
        "products": ["R1T", "R1S"],
    },
    "LCID": {
        "names": ["Lucid Motors", "Lucid"],
        "products": ["Air", "Gravity"],
    },
    "TM": {"names": ["Toyota"]},
    "VOW": {"names": ["Volkswagen", "VW"]},
    "NIO": {"names": ["NIO"]},
    "AAPL": {"names": ["Apple"]},
    "AMZN": {"names": ["Amazon"]},
    "SpaceX": {"names": ["SpaceX"]},
    "X Corp": {"names": ["X Corp", "X.com"]},
    "UBER": {"names": ["Uber"]},
    "Waymo": {"names": ["Waymo"]},
    "MU": {"names": ["Micron"]},
    "TXN": {"names": ["Texas Instruments"]},
    "BBW": {"names": ["Build-A-Bear"]},
}

# Exact sentence text repeated across at least this many articles is treated as
# boilerplate and excluded from sentiment aggregates.
BOILERPLATE_MIN_ARTICLES = 5

# The ALIASES / OTHER_COMPANIES shims were removed; use COMPANIES directly.

# News window (Finnhub free tier ~1yr back — confirm on first pull)
NEWS_START_DATE = "2025-08-01"
NEWS_END_DATE = "2026-08-01"
MIN_USABLE_ARTICLES = 1500  # below -> fallback dataset

LABEL_HORIZONS_DAYS = [1, 3]  # confirm w/ Person B

# --- Entity filter (Goal 2) ---
MIN_SENT_CHARS = 20  # below this, sentences are scraper residue ("Advertisement", "Read more")
SPACY_MODEL = "en_core_web_sm"
SPACY_PIPE_BATCH_SIZE = 50

# --- Coreference resolution -------------------------------------------------
# A HuggingFace id consumed by fastcoref. F-Coref is ~90M params and CPU-viable;
# "biu-nlp/lingmess-coref" is the slower, more accurate drop-in swap.
COREF_MODEL = "biu-nlp/f-coref"
# Subword tokens per inference batch: fastcoref batches by token count, not by
# document count.
COREF_BATCH_SIZE = 10000
# Default for entity_filter.process_articles(use_coref=...). Best-effort: a
# missing backend logs one warning and every sentence is tagged from explicit
# names alone. resolved_by_coref records which rows the model spoke for.
USE_COREF = True
# Clusters keyed by a hash of the exact string coref saw. The most expensive
# stage in the pipeline, so hits matter.
COREF_CACHE_PATH = INTERIM_DATA_DIR / "coref_cache.parquet"

# --- Pipeline input ---------------------------------------------------------
# The cleaned article table the pipeline consumes, one row per article with
# processed_body. Written by notebook 1.2, ~530MB, kept out of git. Under
# data/interim/ because it is a large regenerable intermediate, not a deliverable.
PROCESSED_ARTICLES_PATH = INTERIM_DATA_DIR / "processed_articles.parquet"

# --- Referent verification --------------------------------------------------
# The labelled ground truth a candidate judge is measured against: 270 rows, 100
# with a mention span and 170 without. Load-bearing, and nothing regenerates it.
EVAL_DATA_DIR = DATA_DIR / "eval"
COREF_EVAL_PATH = EVAL_DATA_DIR / "coref_eval_labelled.parquet"
# Verdict cache keyed on (article_id, sent_idx, target, model_id,
# prompt_version). The prompt version is in the key so editing a prompt
# invalidates its verdicts rather than reusing answers formed under the old one.
COREF_JUDGE_CACHE_PATH = INTERIM_DATA_DIR / "coref_judge_cache.parquet"
# The context window the labellers read, in sentences either side. A convention
# to match, not a hyperparameter: a judge shown less is being asked a harder
# question than the labels answer.
EVAL_CONTEXT_PRECEDING = 4
EVAL_CONTEXT_FOLLOWING = 1
# The judge model, 4-bit and CPU-viable. Run through llama-cpp-python rather than
# transformers so it cannot pull a different torch build in underneath FinBERT,
# ABSA and fastcoref. Not tracked; models/ is gitignored.
JUDGE_MODEL_PATH = MODELS_DIR / "gguf" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

# --- Sentiment scoring (Goal 3) ---
FINBERT_MODEL = "ProsusAI/finbert"
MAX_TOKENS = 512
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_CACHE_PATH = INTERIM_DATA_DIR / "finbert_cache.parquet"
LEAD_SENTENCE_WINDOW = 5  # first N sentences counted as the "lead" for sent_entity_lead


# --- Aspect-based sentiment (ABSA) ------------------------------------------
# Scores sentiment toward an aspect term rather than over the whole sentence.
# Runs alongside FinBERT, never instead of it: every absa_* column is parallel to
# a sent_* one, and with ABSA off the absa_* aggregates are NaN.
ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
ABSA_BATCH_SIZE = 16
ABSA_CACHE_PATH = INTERIM_DATA_DIR / "absa_cache.parquet"
USE_ABSA = True

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
