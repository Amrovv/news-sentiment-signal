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

# --- Company registry (single source of truth for entity filtering) ---
#
# The pipeline must work for ANY ticker; TSLA is temporary scaffolding. The
# previous structure split companies into ALIASES (potential targets) and
# OTHER_COMPANIES (a "them" list curated from the TSLA corpus), which is
# target-relative and breaks generalisation: Tesla was absent from
# OTHER_COMPANIES, so with ticker="NVDA" a Tesla sentence got no mentions_other
# tag at all. COMPANIES replaces both with ONE symmetric registry — every entry
# can be the target or an "other" company, selected at runtime by ticker.
#
# Keyed by ticker for public companies, canonical name for private ones.
#
# Tier semantics (see stock_predictor/text/entity_filter.py):
#   "names"       — unambiguous aliases. A match sets mentions_target when the
#                   company is the target, mentions_other otherwise.
#   "person"      — associated people. Ambiguous ("Musk" could mean SpaceX or
#                   X Corp, not Tesla), so a match is tracked as an
#                   informational flag (mentions_ceo) for the TARGET only and
#                   NEVER sets mentions_target. Kept as its own tier per PART 2
#                   of the entity-sentiment spec so the effect of including or
#                   excluding person-linked mentions stays measurable rather
#                   than assumed.
#   "descriptors" — company-descriptor phrases ("the automaker", "the
#                   chipmaker"). These are ANAPHORIC, not explicit: a
#                   descriptor never sets mentions_target/mentions_other by
#                   itself. It resolves, like a generic anaphor, to the most
#                   recently named company whose OWN entry lists that same
#                   descriptor, subject to the ANAPHORA_MAX_GAP decay. That is
#                   why the same phrase is repeated across several entries
#                   below: the set of entries carrying "the automaker" IS the
#                   scope the phrase is allowed to resolve within. Previously
#                   descriptors were hard explicit matches on the target, which
#                   meant "the automaker" in a Ford-heavy article resolved to
#                   Tesla under ticker="TSLA" and, worse, kept its
#                   Tesla-flavoured reading under ticker="F".
#   "products"    — the company's product/model/technology names ("Model Y",
#                   "Full Self-Driving", "Blackwell"). PURELY SUBTRACTIVE: a
#                   product name never sets any mention flag and never becomes
#                   an antecedent. Its job is to stop a product name being
#                   treated as the company itself: entity_filter's
#                   is_substitutable_anaphor() refuses to substitute the
#                   company name over a product surface, so "...whether FSD
#                   would stop for a school bus" is never rewritten into
#                   "...would stop for Tesla". The union of EVERY entry's
#                   products applies on every run, regardless of whose product
#                   it is — a product is never the company under discussion, so
#                   whose it is does not matter.
#                   (This tier had a second consumer, the generic NER
#                   other-company detector, which routinely labelled product
#                   names ORG. That detector was removed; the tier stays for
#                   the substitution guard above.)
#                   These strings used to live in a global NER stoplist, which
#                   put one company's product catalogue in a shared list and
#                   broke the ticker-agnostic constraint: they belong to the
#                   company, so they live on the company's entry.
#
# "person", "descriptors" and "products" are optional per entry; missing means
# empty.
COMPANIES = {
    "TSLA": {
        "names": ["Tesla", "TSLA", "Tesla Inc", "Tesla, Inc.", "$TSLA"],
        "person": ["Musk", "Elon Musk", "Elon"],
        "descriptors": [
            "the automaker",
            "the EV maker",
            "the EV giant",
            "the electric vehicle maker",
        ],
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
        "descriptors": ["the chipmaker"],
        "products": ["Blackwell", "Hopper", "CUDA", "GeForce"],
    },
    "KO": {
        "names": ["Coca-Cola", "Coca Cola", "Coke", "KO", "$KO"],
        "person": [],
        "descriptors": ["the beverage giant"],
    },
    "BYD": {"names": ["BYD"], "descriptors": ["the automaker", "the EV maker"]},
    "F": {"names": ["Ford", "Ford Motor"], "descriptors": ["the automaker"]},
    "GM": {"names": ["General Motors", "GM"], "descriptors": ["the automaker"]},
    "RIVN": {
        "names": ["Rivian"],
        "descriptors": ["the automaker", "the EV maker"],
        "products": ["R1T", "R1S"],
    },
    "LCID": {
        "names": ["Lucid Motors", "Lucid"],
        "descriptors": ["the automaker", "the EV maker"],
        "products": ["Air", "Gravity"],
    },
    "TM": {"names": ["Toyota"], "descriptors": ["the automaker"]},
    "VOW": {"names": ["Volkswagen", "VW"], "descriptors": ["the automaker"]},
    "NIO": {"names": ["NIO"], "descriptors": ["the automaker", "the EV maker"]},
    "AAPL": {"names": ["Apple"]},
    "AMZN": {"names": ["Amazon"]},
    "SpaceX": {"names": ["SpaceX"]},
    "X Corp": {"names": ["X Corp", "X.com"]},
    "UBER": {"names": ["Uber"]},
    "Waymo": {"names": ["Waymo"]},
    "MU": {"names": ["Micron"], "descriptors": ["the chipmaker"]},
    "TXN": {"names": ["Texas Instruments"], "descriptors": ["the chipmaker"]},
    "BBW": {"names": ["Build-A-Bear"]},
}

# Anaphoric references that are generic to *any* company, not ticker-specific.
# A sentence containing one of these and no explicit company name resolves to
# whichever company (target or other) was most recently named — see
# resolve_anaphora() in entity_filter.py.
# Bare "it" used to live here and was a major precision hole: matched anywhere
# in a sentence it is far too promiscuous ("it is worth noting", "make it",
# "it turns out"), so ordinary expletive/pronoun usage silently inflated the
# target-sentiment set. Sentence-initial "It"/"Its" — the genuinely anaphoric
# case — is handled separately in entity_filter (_SENT_INITIAL_IT_RE).
GENERIC_ANAPHORA = ["the company", "the firm"]

# A sentence whose exact text appears in >= this many distinct articles is
# treated as corpus boilerplate (disclosure notices, "Image source: Getty
# Images.", inquiry auto-replies) and excluded from sentiment aggregates.
BOILERPLATE_MIN_ARTICLES = 5

# DEPRECATED — derived views over COMPANIES, kept only so notebook 2.0
# (which imports ALIASES / OTHER_COMPANIES / GENERIC_ANAPHORA) stays
# re-runnable. New code must use COMPANIES directly; these will be removed
# once the notebook is updated.
ALIASES = {
    key: {
        "unambiguous": entry.get("names", []),
        "person": entry.get("person", []),
        "anaphoric": entry.get("descriptors", []),
    }
    for key, entry in COMPANIES.items()
}

# DEPRECATED — derived from COMPANIES (every entry; the target is excluded at
# runtime by key, not by omission from this dict). Kept only for notebook 2.0
# compatibility; new code must use COMPANIES.
OTHER_COMPANIES = {key: entry.get("names", []) for key, entry in COMPANIES.items()}

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

# --- Coreference resolution (replacement for the anaphora heuristic) ---
#
# The heuristic above (most-recently-named company + ANAPHORA_MAX_GAP decay)
# hand-scored at roughly 60% accuracy and supplies about a third of all target
# sentences, so it is the largest known accuracy deficit in the pipeline. A real
# coreference model reads the actual mention chain instead of guessing recency,
# which is what the two known heuristic failure modes need: incidental mentions
# in multi-company roundups, and attribution buried in subordinate clauses.
#
# COREF_MODEL is a HuggingFace id consumed by fastcoref (F-Coref, ~90M params,
# CPU-viable by design; LingMess "biu-nlp/lingmess-coref" is the slower,
# more accurate sibling and is a drop-in swap here).
COREF_MODEL = "biu-nlp/f-coref"
# fastcoref batches by TOKEN COUNT, not by document count: this is the number of
# subword tokens packed into one inference batch.
COREF_BATCH_SIZE = 10000
# Master switch, read as the default of entity_filter.process_articles(
# use_coref=...). FALLBACK BEHAVIOR: coref is strictly best-effort. If the
# fastcoref backend is missing or fails to load, coref.is_available() returns
# False, one warning is logged, and every sentence falls back to the existing
# anaphora heuristic — the pipeline never hard-fails because coref is
# unavailable, and it never silently changes tag semantics: the two mechanisms
# stay separately measurable via resolved_by_coref vs resolved_by_anaphora.
USE_COREF = True
# On-disk cache for coreference clusters, mirroring SENTIMENT_CACHE_PATH.
# Coref is deterministic per document text and is the most expensive stage in
# the text pipeline (~600s over the corpus), so every document's clusters are
# keyed by a hash of the exact string coref saw and reused across runs.
COREF_CACHE_PATH = INTERIM_DATA_DIR / "coref_cache.parquet"

# --- Coreference referent verification (coref_eval.py) ---
# The hand-labelled ground truth for "did coref resolve this anaphor to the
# right company". 270 rows (100 with an anaphor span, 170 without), built and
# documented in notebook 2.7. It is the ONLY thing a candidate verification
# judge is measured against -- two earlier attempts were argued into the
# pipeline without being measured and both failed, so this file is load-bearing.
EVAL_DATA_DIR = DATA_DIR / "eval"
COREF_EVAL_PATH = EVAL_DATA_DIR / "coref_eval_labelled.parquet"
# Verdict cache for referent-verification judges, keyed on
# (article_id, sent_idx, target, model_id, prompt_version). prompt_version is in
# the key deliberately: editing a prompt must invalidate its verdicts rather
# than silently reuse answers formed under the old wording.
COREF_JUDGE_CACHE_PATH = INTERIM_DATA_DIR / "coref_judge_cache.parquet"
# The context window the human labellers actually read, in sentences either side
# of the one being judged. A judge shown less than this is being asked a harder
# question than the labels answer, so these are a convention to match, not a
# hyperparameter to tune.
EVAL_CONTEXT_PRECEDING = 4
EVAL_CONTEXT_FOLLOWING = 1
# The local instruct model that acts as the referent-verification judge.
# Qwen2.5-7B-Instruct at 4-bit (Q4_K_M, ~4.7GB) runs on CPU on this machine in
# ~16GB RAM; torch here is 2.13.0+cpu with no CUDA, and llama-cpp-python is
# deliberately used instead of transformers because it is independent of torch
# and cannot drag a different build in underneath FinBERT/ABSA/fastcoref.
# Not tracked in git (models/ is gitignored); see notebook 2.7 §11 for the fetch.
JUDGE_MODEL_PATH = MODELS_DIR / "gguf" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

# --- Sentiment scoring (Goal 3) ---
FINBERT_MODEL = "ProsusAI/finbert"
MAX_TOKENS = 512
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_CACHE_PATH = INTERIM_DATA_DIR / "finbert_cache.parquet"
LEAD_SENTENCE_WINDOW = 5  # first N sentences counted as the "lead" for sent_entity_lead

# Master switch for the anaphora recency-heuristic FALLBACK (scoped descriptors
# and generic anaphora/sentence-initial "It" — see
# entity_filter.resolve_anaphora()). Read as the default of
# entity_filter.process_articles(use_anaphora_fallback=...).
#
# A context-aware audit of 100 anaphora-resolved sentences (auditor saw a
# 6-sentence window and stated the referent independently; the passage was
# judged determinate in 99% of rows) found the heuristic resolved the correct
# referent on only 13 of 100 -- coreference, run on the equivalent 100-sentence
# coref sample, scored 74%. Of the heuristic's failures, 61% resolved to a
# DIFFERENT company actually named in the passage (typical case: "Pilot is the
# largest network of travel centers in North America... The company serves an
# average of 1.2 million guests per day" tagged Tesla). Disabling it costs 278
# non-boilerplate target sentences (1.9% of the target set) -- the heuristic
# only ever runs where coref (precedence 2) already produced nothing, so that
# is the entire size of the gap it was filling.
#
# The capability is retained behind this flag rather than deleted: if coref
# coverage ever regresses (backend unavailable, model swapped, etc.), it can be
# re-enabled without restoring code.
USE_ANAPHORA_FALLBACK = False

# --- Aspect-based sentiment (ABSA) ---
#
# FinBERT scores a SENTENCE, not a target. "Build-A-Bear outperformed Tesla"
# is a positive sentence, and every target-sentence aggregate reads it as
# positive news about Tesla. An aspect-based model is handed the aspect term
# explicitly and scores sentiment TOWARD that aspect, which is the shape of the
# question the pipeline actually asks.
#
# This runs ALONGSIDE FinBERT, never instead of it: every absa_* feature is a
# parallel column next to its sent_* FinBERT counterpart so the two can be
# compared on the same corpus before anything is decided. ABSA off (or its
# score columns absent) must leave the pipeline working, with the absa_*
# aggregates all NaN.
#
# The model is SemEval/review-trained rather than finance-trained, which is the
# obvious risk: it may read the target correctly on comparatives and worse than
# FinBERT on ordinary financial prose. That trade is a measurement, not an
# assumption — hence both families of columns.
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
