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
#                   an antecedent. Its only job is to suppress the generic NER
#                   other-company detector, which routinely labels product
#                   names ORG ("Tesla ... : Optimus humanoids, FSD, Robotaxi,
#                   Cybercab, Megapack") and would otherwise invent an
#                   other-company mention out of the TARGET's own catalogue.
#                   The union of EVERY entry's products is suppressed for every
#                   run, regardless of whose product it is — a product is never
#                   a company under discussion, so whose it is does not matter.
#                   These strings used to live in NER_ORG_STOPLIST, which put
#                   one company's product catalogue in a global list and broke
#                   the ticker-agnostic constraint: they belong to the company,
#                   so they live on the company's entry.
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

# ORG entities spaCy's NER finds that are NOT "another company under
# discussion" in the sense entity_filter cares about. The curated COMPANIES
# registry cannot keep up with the tail of company names that show up in a news
# corpus (Avidity Biosciences, Samsung, ...), so entity_filter also treats any
# ORG entity the registry does not know as an other-company mention. That
# generic rule needs a subtractive filter, because spaCy labels exchanges,
# newswires and government bodies ORG too, and those are the *venue* or the
# *source* of a story rather than a rival whose sentiment would pollute the
# target's.
#
# Entries are compared against entity_filter's normalized form of the entity
# text: leading "The " removed, trailing punctuation stripped, casefolded — so
# every entry here must already be lowercase.
#
# NECESSARILY INCOMPLETE. This list was tuned by eyeballing the ORG entities
# spaCy actually produced on THIS corpus (a year of TSLA/NVDA financial news);
# it is a precision patch on a generic detector, not an ontology. A different
# corpus will surface different non-company ORGs and will want additions here.
NER_ORG_STOPLIST = frozenset(
    {
        # Exchanges and indices — the venue a stock trades on, not a rival.
        "nasdaq",
        "nyse",
        "s&p",
        "s&p 500",
        "dow",
        "dow jones",
        "russell 2000",
        # Media and research outlets — the source of the story, not its subject.
        # These are extremely frequent (bylines, "according to", disclosures),
        # so leaving them in would swamp the genuine NER signal.
        "reuters",
        "bloomberg",
        "cnbc",
        "benzinga",
        "the motley fool",
        "motley fool",
        "yahoo finance",
        "marketwatch",
        "barron's",
        "seeking alpha",
        "investing.com",
        "tipranks",
        "zacks",
        # Institutions and regulators. Both the acronym and the full form are
        # listed: only the acronyms used to be here, so "Tesla's FSD technology
        # also faces scrutiny from the National Highway Traffic Safety
        # Administration (NHTSA)" was tagged as naming another COMPANY.
        "sec",
        "securities and exchange commission",
        "the fed",
        "federal reserve",
        "congress",
        "eu",
        "european union",
        "nhtsa",
        "national highway traffic safety administration",
        "faa",
        "federal aviation administration",
        "irs",
        "internal revenue service",
        "doj",
        "department of justice",
        "department of transportation",
        # Generic tokens spaCy mislabels as ORG.
        "wall street",
        "main street",
        "ev",
        "ai",
        "community",
        # Zacks-style rating/metric words. Syndicated Zacks copy writes them
        # capitalised mid-sentence ("Tesla scores well on the Momentum metric,
        # while offering satisfactory Growth and Quality, but poor Value"),
        # which spaCy reads as ORG.
        "momentum",
        "growth",
        "quality",
        "value",
        "vgm",
        "style scores",
        # Press-release wires — the distribution channel, not a subject.
        "business wire",
        "pr newswire",
        "globe newswire",
        "accesswire",
    }
    | {
        # --- measured additions ---------------------------------------------
        # Everything above was written from prior knowledge of the corpus.
        # These were added after counting the ORG keys the generic detector
        # ACTUALLY produced over all 71k tagged sentences: they are the
        # non-company entries in the top ~60 by frequency. Kept as a separate
        # block so the hand-written list stays legible and this one stays
        # obviously revertible/extendable as the corpus changes.
        #
        # Financial acronyms and reporting jargon spaCy reads as ORG.
        "ipo",
        "eps",
        "yoy",
        "etf",
        "llc",
        "av",
        "q1",
        "q2",
        "q3",
        "q4",
        "invest",
        # (The product/technology names that used to sit here — fsd,
        # full self-driving, autopilot, robotaxi — moved to COMPANIES["TSLA"]
        # ["products"]. A global stoplist must not carry one company's product
        # catalogue; see the "products" tier note above.)
        # More outlets, data vendors and syndication furniture. The .com forms
        # are separate keys because normalization keeps the domain.
        "benzinga.com",
        "benzinga news",
        "shutterstock",
        "shutterstock © 2026 benzinga.com",
        "getty images",
        "google news",
        "forbes",
        "stocktwits",
        "gurufocus",
        "wall street journal",
        "the wall street journal",
        "zacks investment research",
        "zacks consensus estimate",
        "free stock analysis report",
        "www.sipc.org",
        # Institutions, regulators and index/ETF proxies.
        "fed",
        "treasury",
        "nasa",
        "finra",
        "spy",
        "qqq",
        "invesco qqq trust",
        "mag 7",
        # Frequent PERSON mislabelled as ORG in this corpus.
        "trump",
    }
)

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

# --- Sentiment scoring (Goal 3) ---
FINBERT_MODEL = "ProsusAI/finbert"
MAX_TOKENS = 512
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_CACHE_PATH = INTERIM_DATA_DIR / "finbert_cache.parquet"
LEAD_SENTENCE_WINDOW = 5  # first N sentences counted as the "lead" for sent_entity_lead

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
