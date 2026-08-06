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

# Entity-filter aliases
ALIASES = {
    "TSLA": ["Tesla", "TSLA", "Musk", "Elon Musk", "the automaker", "the EV maker"],
    "NVDA": ["Nvidia", "NVDA", "Jensen Huang", "the chipmaker"],
    "KO": ["Coca-Cola", "Coca Cola", "Coke", "KO"],
}

# News window (Finnhub free tier ~1yr back — confirm on first pull)
NEWS_START_DATE = "2025-08-01"
NEWS_END_DATE = "2026-08-01"
MIN_USABLE_ARTICLES = 1500  # below -> fallback dataset

LABEL_HORIZONS_DAYS = [1, 3]  # confirm w/ Person B

FINBERT_MODEL = "ProsusAI/finbert"
MAX_TOKENS = 512

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
