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

# Figures written by notebooks, which name their own files after the notebook
# that drew them (3.4-relative-sentiment-per-fold.png). Generated reports do not
# write here -- each one keeps its figures beside itself, see report_dir().
FIGURES_DIR = REPORTS_DIR / "figures"

# Reports are grouped by the pipeline that generates them, one directory each,
# because a flat reports/ becomes unreadable once four tickers each contribute a
# fetch, market and merge report plus their figures. A section owns its figures:
#
#     reports/fetch/TSLA_raw_fetch_report.md
#     reports/fetch/figures/TSLA_raw_fetch_daily.png
#
# so a report and the images it links move, or are deleted, as one unit.
REPORT_SECTIONS = ["fetch", "market", "merge", "text", "findings"]


def report_dir(section: str) -> Path:
    """Directory for one pipeline's reports, created on demand."""
    if section not in REPORT_SECTIONS:
        raise ValueError(f"Unknown report section {section!r}; expected one of {REPORT_SECTIONS}")
    path = REPORTS_DIR / section
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_figures_dir(section: str) -> Path:
    """Where one pipeline's report figures live, created on demand.

    Reports link these relatively, as `figures/{name}.png`, so a report renders
    on GitHub, in an editor preview, and after the whole directory is moved.
    """
    path = report_dir(section) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        "names": ["Nvidia", "NVDA", "Nvidia Corp", "Nvidia Corporation", "$NVDA"],
        "person": ["Jensen Huang", "Huang", "Jensen"],
        "products": [
            "Blackwell",
            "Hopper",
            "Ada Lovelace",
            "Ampere",
            "GeForce",
            "RTX",
            "GTX",
            "CUDA",
            "Omniverse",
            "DGX",
            "Grace",
            "NVLink",
            "Tensor Core",
            "DLSS",
            "Jetson",
            "Quadro",
            "Shield TV",
            "Drive",
        ],
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
    "AAPL": {
        "names": ["Apple", "AAPL", "Apple Inc", "Apple Inc.", "$AAPL"],
        # "Cook" and "Jobs" alone are excluded: common words ("to cook", "jobs" the
        # noun), unlike "Musk"/"Huang" -- would false-positive constantly. Full
        # names only.
        "person": ["Tim Cook", "Steve Jobs"],
        "products": [
            "iPhone",
            "iPad",
            "Mac",
            "MacBook",
            "MacBook Air",
            "MacBook Pro",
            "iMac",
            "Apple Watch",
            "AirPods",
            "Vision Pro",
            "iOS",
            "macOS",
            "iPadOS",
            "App Store",
            "Siri",
            "Apple Intelligence",
            "Apple TV",
            "HomePod",
        ],
    },
    "AMZN": {
        "names": ["Amazon", "AMZN", "Amazon.com", "Amazon Inc", "Amazon Inc.", "$AMZN"],
        "person": ["Andy Jassy", "Jassy", "Jeff Bezos", "Bezos"],
        "products": [
            "AWS",
            "Amazon Web Services",
            "Prime",
            "Amazon Prime",
            "Prime Video",
            "Alexa",
            "Kindle",
            "Echo",
            "Fire TV",
            "Whole Foods",
            "Amazon Music",
            "Ring",
            "Amazon Go",
            "Kuiper",
            "Bedrock",
        ],
    },
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

# --- Article fetch (Goal 1) --------------------------------------------------
# Raw Finnhub pull, before the scrape stage resolves source/time/body. This
# single-file constant is what the existing TSLA-only run_pipeline.py reads;
# raw_articles_path()/processed_articles_path() below are the per-ticker forms
# the fetch package's own CLIs (finnhub_pull.py, scrape.py) read and write.
RAW_ARTICLES_PATH = RAW_DATA_DIR / "raw_articles.parquet"


def raw_articles_path(ticker: str) -> Path:
    """Where finnhub_pull.py's CLI writes one ticker's raw pull."""
    return RAW_DATA_DIR / f"{ticker}_raw_articles.parquet"


def processed_articles_path(ticker: str) -> Path:
    """Where scrape.py's CLI writes one ticker's scraped, cleaned corpus."""
    return INTERIM_DATA_DIR / f"{ticker}_processed_articles.parquet"


def processed_chunk_dir(ticker: str) -> Path:
    """Where scrape.py's chunked run checkpoints each chunk's result."""
    return INTERIM_DATA_DIR / f"{ticker}_processed_chunks"


# `company_news` caps results per call regardless of window width and fills
# most-recent-first (notebooks/modelling/3.0, section 6, diagnosed this as the
# cause of the original pull's tail-of-month bursts). This is a conservative
# early-warning line, not a confirmed cap value: pull_company_news logs a
# warning if any single call's count reaches it, since that call may be
# getting silently truncated the same way. The fetch report is what actually
# confirms whether a given pull is steady, this is just a tripwire during it.
FETCH_CAP_WARN_COUNT = 100

# finnhub-python's client has no retry/backoff of its own (a bare
# requests.Session, no HTTPAdapter/Retry mounted), and a daily-windowed pull
# is ~365 calls where the old monthly one was ~13 -- one transient timeout
# used to be rare enough to ignore, now it's close to guaranteed over a full
# year. pull_company_news retries a failing window this many times, waiting
# FETCH_RETRY_BACKOFF_SECONDS between attempts, before logging it as failed
# and moving on rather than losing the whole run.
FETCH_RETRY_ATTEMPTS = 3
FETCH_RETRY_BACKOFF_SECONDS = 5

# Sources found readable by the probe in notebooks/text/1.1-aw-scraper-probe.ipynb:
# most requests reach a 200 and the body is long enough to be a real article.
# SeekingAlpha, ChartMill, MarketWatch (blocked outright) and CNBC, Finnhub
# (reach but no real body) are excluded.
OPEN_SOURCES = ["Yahoo", "Benzinga", "DowJones"]

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
SCRAPE_MAX_WORKERS = 8
SCRAPE_TIMEOUT = 12  # seconds
MIN_BODY_CHARS = 500  # cleaned body shorter than this is a stub, cookie wall, or paywall teaser
MAX_SHIFT_HOURS = (
    6  # a scraped time this far from the Finnhub API time is treated as a repost, not a correction
)

# A cross-host canonical is normally trusted outright (it's the original
# outlet's own page), but that has no floor against a genuinely old,
# re-referenced story surfacing in Finnhub's feed years after it first ran --
# found in practice: a 2023 CNN article scored into a 2025-2026 pull. Cap how
# far a cross-host date may diverge from the Finnhub API time before it's
# rejected back to the API time instead of trusted.
MAX_CROSS_HOST_SHIFT_DAYS = 30
REQ_PER_SEC = 3.5  # global scrape rate, held under Yahoo's 429 limit

# A full corpus can be tens of thousands of articles, and at REQ_PER_SEC that
# is well over an hour of continuous network I/O -- a long enough window that
# a killed process, a lost connection, or a machine restart partway through
# is a real risk, not a hypothetical one. scrape.py's CLI runs in chunks of
# this size, checkpointing each chunk's result to processed_chunk_dir()
# before starting the next, so an interruption costs at most one chunk's
# worth of work rather than the whole run, and a restart skips any chunk
# whose checkpoint already exists.
SCRAPE_CHUNK_SIZE = 500

# The 3-day horizon was collected and examined, then dropped: notebook 3.0's
# EDA found its columns carry close to no signal, and 3.1 excluded them from
# the model-ready table on that basis. Building them anyway would put two
# forward-looking columns in the table that nothing is allowed to use, which is
# a leak waiting to happen rather than an option kept open. Re-add 3 here to
# rebuild them.
LABEL_HORIZONS_DAYS = [1]

# --- Market layer ------------------------------------------------------------
# Price history and the NYSE calendar are shared across tickers, so they live in
# one file each: yfinance returns every symbol in a single frame and the trading
# calendar is the same for all of them. Earnings dates are per-company, so they
# follow the fetch layer's `{TICKER}_` convention instead.
RAW_OHLCV_PATH = RAW_DATA_DIR / "raw_ohlcv.parquet"
RAW_SCHEDULE_PATH = RAW_DATA_DIR / "raw_schedule.parquet"


def raw_earnings_path(ticker: str) -> Path:
    """Where prices.py's CLI writes one ticker's earnings calendar."""
    return RAW_DATA_DIR / f"{ticker}_raw_earnings.parquet"


def merged_dir(ticker: str) -> Path:
    """Where merge.run_pipeline writes one ticker's joined table.

    The pooled table sits beside these directories rather than inside one, at
    `data/processed/merged/pooled.parquet`, since it belongs to no single ticker.
    """
    return PROCESSED_DATA_DIR / "merged" / ticker


def market_run_dir(ticker: str) -> Path:
    """Where market.run_pipeline writes one ticker's deliverable and its data dictionary.

    Mirrors the text layer's data/processed/pipeline_run/{TICKER}/, so the two
    tables a model merges sit at symmetric paths.
    """
    return PROCESSED_DATA_DIR / "market_run" / ticker


# Every pre-publication feature needs history *before* the first article, and
# the 3-day label needs bars *after* the last one. Both windows are widened by
# this much beyond the news window: 20 trading days of lookback is ~28 calendar
# days, and 40 leaves margin for holidays.
PRICE_PAD_DAYS = 40
SCHEDULE_PAD_DAYS = 10

# Feature windows. Changing one changes the column it names, not its meaning:
# momentum_5d is defined as "cumulative return over the 5 trading days before
# publication", so the column name is generated from this list.
MOMENTUM_LOOKBACKS = [1, 5, 20]
VOLATILITY_WINDOW = 20
BETA_WINDOW = 20
RELATIVE_VOLUME_WINDOW = 20

# news_volume counts the target's own prior articles in this many days. Scoped
# to one ticker's corpus, since the pipeline runs per ticker: it measures how
# noisy coverage of *this* company was going into the article, not how busy the
# market was overall.
NEWS_VOLUME_LOOKBACK_DAYS = 3

# --- Entity filter (Goal 2) ---
MIN_SENT_CHARS = 20  # below this, sentences are scraper residue ("Advertisement", "Read more")
SPACY_MODEL = "en_core_web_sm"
SPACY_PIPE_BATCH_SIZE = 200

# Pipeline components loaded but never read. entity_filter needs the parser
# (doc.sents, token.dep_, token.head) and the NER (doc.ents); nothing anywhere
# reads pos_, tag_, morph or lemma_, which is all the tagger, attribute_ruler
# and lemmatizer produce. Both the parser and the NER listen to tok2vec rather
# than to the tagger, so dropping these does not change what they predict.
# DO NOT add "ner" or "parser" here -- see the warning on _get_nlp().
SPACY_EXCLUDE = ["lemmatizer", "tagger", "attribute_ruler"]

# Worker processes for the sentence-splitting parse, the pipeline's largest
# CPU-bound cost (~11.5 min for 12k articles single-process). Workers spawn on
# Windows, so the caller must sit under an `if __name__ == "__main__"` guard --
# run_pipeline does. Each worker holds its own copy of the model and the Docs in
# flight, so this buys wall time with memory; 4 of 6 cores leaves room for both.
SPACY_N_PROCESS = 4

# Below this many documents the spawn cost outweighs the parallelism, so short
# batches (tests, notebooks, a handful of articles) keep the single-process path.
SPACY_MULTIPROCESS_MIN_DOCS = 500

# --- Coreference resolution -------------------------------------------------
# A HuggingFace id consumed by fastcoref. F-Coref is ~90M params and CPU-viable;
# "biu-nlp/lingmess-coref" is the slower, more accurate drop-in swap.
COREF_MODEL = "biu-nlp/f-coref"
# Subword tokens per inference batch: fastcoref batches by token count, not by
# document count. The CPU figure is bounded by system RAM and can be generous;
# the GPU one cannot, because a coref model scores every candidate span pair and
# so grows far faster than linearly in batch tokens. 10000 tokens on an 8GB card
# shared with a desktop OOMs outright, hence a much smaller GPU batch rather
# than the same number on both.
COREF_BATCH_SIZE = 10000
COREF_BATCH_SIZE_GPU = 1500

# Documents per predict() call. fastcoref holds every prediction it has made
# until the call returns, and on a GPU those carry device tensors, so one call
# over a whole corpus grows until the card is full regardless of how small
# max_tokens_in_batch is. Chunking bounds that, and gives the cache something to
# save partway through a long resolve rather than only at the end.
COREF_DOC_CHUNK = 500
# Default for entity_filter.process_articles(use_coref=...). Best-effort: a
# missing backend logs one warning and every sentence is tagged from explicit
# names alone. resolved_by_coref records which rows the model spoke for.
USE_COREF = True
# Clusters keyed by a hash of the exact string coref saw. The most expensive
# stage in the pipeline, so hits matter.
COREF_CACHE_PATH = INTERIM_DATA_DIR / "coref_cache.parquet"

# --- Pipeline input ---------------------------------------------------------
# The cleaned article table the pipeline consumes, one row per article with
# processed_body. Written by stock_predictor.fetch.scrape, ~530MB, kept out
# of git. Under data/interim/ because it is a large regenerable intermediate,
# not a deliverable.
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

# GPU layers to offload for the judge model, when a CUDA-capable llama-cpp-python
# build is installed: -1 offloads every layer (fastest, tried first), 0 forces
# CPU-only (the original, always-viable path this project was built around).
# CPU-viable was a real constraint, not a preference -- keep 0 working.
JUDGE_N_GPU_LAYERS = -1

# Batch sizes and flash attention for the GPU path. Measured on an RTX 2070
# Super, real judge prompts via the actual pipeline on the true unmodified
# CPU-only build: 5.5s/row (CPU) to 0.20-0.22s/row at these settings, ~27x.
# Both are ignored (harmlessly) when running CPU-only.
JUDGE_N_BATCH = 1024
JUDGE_N_UBATCH = 1024
JUDGE_FLASH_ATTN = True

# Quantized KV cache type for the GPU path (ggml type id: 8 = q8_0). Cuts KV
# cache memory-bandwidth cost a bit further; negligible accuracy risk for a
# single-word yes/no/unsure classification. None leaves llama.cpp's default
# (f16) in place -- the safe fallback if this ever needs disabling.
JUDGE_KV_CACHE_TYPE = 8

# --- Torch model placement ---------------------------------------------------
# Device for the three torch models (FinBERT, ABSA, fastcoref); the GGUF judge
# picks its own via JUDGE_N_GPU_LAYERS above. "auto" uses a CUDA device when
# torch can see one and CPU otherwise; "cpu" or "cuda" pin it explicitly.
# Resolved once per process by stock_predictor.text.device, which also falls
# back to CPU per model if a move fails, so CPU-only stays a working setup.
TEXT_DEVICE = "auto"

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
