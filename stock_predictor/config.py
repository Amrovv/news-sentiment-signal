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
CONTROL_TICKER = "KO"        # low-news control
MARKET_INDEX = "SPY"         # abnormal-return benchmark

# Entity-filter aliases, tiered by how unambiguous a match is (see
# stock_predictor/text/entity_filter.py). Kept separate per PART 2 of the
# entity-sentiment spec so that ambiguous person-linked mentions (e.g. "Musk"
# could mean SpaceX/X) can be included or excluded and the effect measured,
# rather than assumed.
ALIASES = {
    "TSLA": {
        "unambiguous": ["Tesla", "TSLA", "Tesla Inc", "Tesla, Inc.", "$TSLA"],
        "person": ["Musk", "Elon Musk", "Elon"],
        "anaphoric": ["the automaker", "the EV maker"],
    },
    "NVDA": {
        "unambiguous": ["Nvidia", "NVDA", "Nvidia Corp", "$NVDA"],
        "person": ["Jensen Huang", "Huang"],
        "anaphoric": ["the chipmaker"],
    },
    "KO": {
        "unambiguous": ["Coca-Cola", "Coca Cola", "Coke", "KO", "$KO"],
        "person": [],
        "anaphoric": ["the beverage giant"],
    },
}

# Anaphoric references that are generic to *any* company, not ticker-specific.
# A sentence containing one of these and no explicit company name resolves to
# whichever company (target or other) was most recently named — see
# resolve_anaphora() in entity_filter.py.
GENERIC_ANAPHORA = ["the company", "the firm", "it"]

# Other companies to recognise for the "other-company" contrast tag and for
# anaphora tracking (so "the company" after a BYD sentence resolves to BYD,
# not TSLA). Not exhaustive — covers names that actually appear often in the
# scraped TSLA corpus (see notebooks/a_entity_sentiment.ipynb section 2.2).
OTHER_COMPANIES = {
    "BYD": ["BYD"],
    "Ford": ["Ford", "Ford Motor"],
    "GM": ["General Motors", "GM"],
    "Rivian": ["Rivian"],
    "Lucid": ["Lucid Motors", "Lucid"],
    "Toyota": ["Toyota"],
    "Volkswagen": ["Volkswagen", "VW"],
    "NIO": ["NIO"],
    "Nvidia": ["Nvidia"],
    "Apple": ["Apple"],
    "Amazon": ["Amazon"],
    "SpaceX": ["SpaceX"],
    "X Corp": ["X Corp", "X.com"],
}

# News window (Finnhub free tier ~1yr back — confirm on first pull)
NEWS_START_DATE = "2025-08-01"
NEWS_END_DATE = "2026-08-01"
MIN_USABLE_ARTICLES = 1500  # below -> fallback dataset

LABEL_HORIZONS_DAYS = [1, 3]  # confirm w/ Person B

# --- Entity filter (Goal 2) ---
MIN_SENT_CHARS = 20  # below this, sentences are scraper residue ("Advertisement", "Read more")
SPACY_MODEL = "en_core_web_sm"
SPACY_PIPE_BATCH_SIZE = 50
# Anaphora resolution decays after this many sentences without a fresh explicit
# company mention -- without a decay window, "the company"/"it" late in a long
# full-body article keeps resolving to whatever was named many paragraphs
# earlier, even after the topic has drifted to unrelated content (author bios,
# boilerplate, tangential discussion). See notebooks/a_entity_sentiment.ipynb
# section 4.3 for the hand-checked accuracy delta this produced.
ANAPHORA_MAX_GAP = 6

# --- Sentiment scoring (Goal 3) ---
FINBERT_MODEL = "ProsusAI/finbert"
MAX_TOKENS = 512
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_CACHE_PATH = INTERIM_DATA_DIR / "finbert_cache.parquet"
LEAD_SENTENCE_WINDOW = 5  # first N sentences counted as the "lead" for sent_entity_lead

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
