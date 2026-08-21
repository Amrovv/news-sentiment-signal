"""FinBERT sentiment scoring and article-level aggregation.

Runs downstream of entity_filter, on the SENTENCE_COLUMNS table.

    score_sentences()            batched, on-disk-cached FinBERT scorer
    score_sentence_table()       scores a corpus table, merges pos/neg/neu on
    score_headlines()            headlines standalone, same cache
    aggregate_article_features() per-article aggregates from a scored table
    build_model_features()       the lean model-facing table
    analyze()                    single-article path for the live demo

needs_score() decides which sentences are scored, and the aggregator reads the
same predicate, so the two cannot drift.

Cache saves MERGE with the cache loaded at the start of the call. score_sentences()
returns only the current call's hashes, so saving it alone truncates the file.

Scores are stored as raw pos/neg/neu triples. The fus_* columns are the exception,
being signed scalars from fusion.py; the triples behind them are still persisted.
"""

import hashlib

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from stock_predictor.config import (
    FINBERT_MODEL,
    LEAD_SENTENCE_WINDOW,
    PRIMARY_TICKER,
    MAX_TOKENS,
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_CACHE_PATH,
)
from stock_predictor.text import entity_filter, fusion

CACHE_COLUMNS = ["text_hash", "pos", "neg", "neu"]

# NaN when ABSA is absent; no sent_* column changes because these exist.
ABSA_SCORE_COLUMNS = ["absa_pos", "absa_neg", "absa_neu"]
ABSA_FEATURE_COLUMNS = [
    "absa_entity_pos",
    "absa_entity_neg",
    "absa_entity_neu",
    "absa_ceo_pos",
    "absa_ceo_neg",
    "absa_ceo_neu",
]

# NaN when fusion cannot be computed, which is whenever the ABSA scores are absent.
FUSION_FEATURE_COLUMNS = [
    f"fus_{variant}_{agg}"
    for variant in fusion.AGGREGATED_VARIANTS
    for agg in fusion.FUSION_AGGREGATIONS
]

# Sentence flags that put a sentence into a bucket some aggregate reads.
# See needs_score() -- this list is the enumeration behind that predicate.
CONSUMED_MENTION_COLUMNS = ["mentions_target", "mentions_ceo"]


def needs_score(sentences_df: pd.DataFrame) -> pd.Series:
    """Boolean mask of sentences whose scores are consumed downstream.

        (mentions_target | mentions_ceo) & ~is_boilerplate

    Shared by the scorer and the aggregator so the two cannot drift: a feature reading
    a new sentence bucket must be added here or it gets NaN for every article.

    A missing is_boilerplate defaults to all-False; the mentions_* columns are
    required. Returns a bool Series aligned to sentences_df.index.
    """
    missing = [c for c in CONSUMED_MENTION_COLUMNS if c not in sentences_df.columns]
    if missing:
        raise ValueError(
            f"needs_score: sentences_df is missing required column(s): {missing}. "
            f"Expected the entity_filter sentence schema "
            f"(columns {CONSUMED_MENTION_COLUMNS})."
        )

    if len(sentences_df) == 0:
        return pd.Series(dtype=bool, index=sentences_df.index)

    in_bucket = pd.Series(False, index=sentences_df.index)
    for col in CONSUMED_MENTION_COLUMNS:
        in_bucket |= sentences_df[col].fillna(False).astype(bool)

    if "is_boilerplate" in sentences_df.columns:
        boiler = sentences_df["is_boilerplate"].fillna(False).astype(bool)
    else:
        boiler = pd.Series(False, index=sentences_df.index)

    return (in_bucket & ~boiler).astype(bool)


def hash_text(text: str) -> str:
    """sha256 of the cleaned/stripped text, first 16 hex chars. Cache key."""
    cleaned = (text or "").strip()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]


def load_cache(path=SENTIMENT_CACHE_PATH) -> pd.DataFrame:
    """Load the on-disk FinBERT score cache. Returns an empty frame with the
    expected columns (text_hash, pos, neg, neu) if the file doesn't exist."""
    if not path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    df = pd.read_parquet(path)
    missing = set(CACHE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Sentiment cache at {path} is missing columns: {missing}")
    return df[CACHE_COLUMNS]


def save_cache(cache_df: pd.DataFrame, path=SENTIMENT_CACHE_PATH) -> None:
    """Persist the cache, deduped on text_hash (keep last). Creates the
    parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped = cache_df.drop_duplicates(subset="text_hash", keep="last")[CACHE_COLUMNS]
    deduped.to_parquet(path, index=False)
    logger.info(f"Saved sentiment cache with {len(deduped)} entries to {path}")


def _load_model_and_tokenizer():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info(f"Loading FinBERT model '{FINBERT_MODEL}'")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    return model, tokenizer


def _build_label_index_map(model) -> dict[str, int]:
    """Map {"positive": idx, "negative": idx, "neutral": idx} by reading
    model.config.id2label rather than assuming FinBERT's documented order
    ({0: positive, 1: negative, 2: neutral}). Verifies all three expected
    labels are present regardless of case/order in the config."""
    id2label = model.config.id2label
    index_map: dict[str, int] = {}
    for idx, label in id2label.items():
        norm = str(label).strip().lower()
        if norm in ("positive", "negative", "neutral"):
            index_map[norm] = int(idx)
    expected = {"positive", "negative", "neutral"}
    if set(index_map.keys()) != expected:
        raise ValueError(
            f"Unexpected FinBERT id2label mapping {id2label!r}; "
            f"could not resolve {expected} labels."
        )
    return index_map


def score_sentences(
    texts: list[str],
    model=None,
    tokenizer=None,
    cache_df: pd.DataFrame | None = None,
    batch_size: int = SENTIMENT_BATCH_SIZE,
) -> pd.DataFrame:
    """Core batched + cached FinBERT scorer.

    Dedupes on text_hash, splits into cache hits and a needs-scoring remainder, sorts
    that remainder by length before batching so same-batch sequences pad alike (20-30%
    faster on CPU), then runs under torch.no_grad() truncated to MAX_TOKENS.

    Label order is read from model.config.id2label, not assumed. If every hash hits,
    the model is never loaded even with model/tokenizer None.

    Returns one row per unique text_hash. Does not write to disk.
    """
    if cache_df is None or len(cache_df) == 0:
        cache_lookup: dict[str, tuple[float, float, float]] = {}
    else:
        cache_lookup = {
            row.text_hash: (row.pos, row.neg, row.neu) for row in cache_df.itertuples(index=False)
        }

    # Unique texts, keeping the first occurrence of each hash.
    unique_by_hash: dict[str, str] = {}
    for t in texts:
        h = hash_text(t)
        if h not in unique_by_hash:
            unique_by_hash[h] = t

    cached_rows = []
    to_score: list[tuple[str, str]] = []  # (hash, text)
    for h, t in unique_by_hash.items():
        if h in cache_lookup:
            pos, neg, neu = cache_lookup[h]
            cached_rows.append({"text_hash": h, "pos": pos, "neg": neg, "neu": neu})
        else:
            to_score.append((h, t))

    logger.info(
        f"score_sentences: {len(unique_by_hash)} unique texts "
        f"({len(cached_rows)} cache hits, {len(to_score)} to score)"
    )

    if not to_score:
        return pd.DataFrame(cached_rows, columns=CACHE_COLUMNS)

    if model is None or tokenizer is None:
        loaded_model, loaded_tokenizer = _load_model_and_tokenizer()
        model = model or loaded_model
        tokenizer = tokenizer or loaded_tokenizer

    label_idx = _build_label_index_map(model)
    pos_i, neg_i, neu_i = label_idx["positive"], label_idx["negative"], label_idx["neutral"]

    # Sort needs-scoring subset by character length (cheap proxy for token
    # length) before batching to reduce padding waste.
    to_score.sort(key=lambda pair: len(pair[1]))

    scored_rows = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(to_score), batch_size), desc="Scoring sentences (FinBERT)"
        ):
            batch = to_score[start : start + batch_size]
            batch_hashes = [h for h, _ in batch]
            batch_texts = [t for _, t in batch]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

            for h, row in zip(batch_hashes, probs.tolist()):
                scored_rows.append(
                    {
                        "text_hash": h,
                        "pos": row[pos_i],
                        "neg": row[neg_i],
                        "neu": row[neu_i],
                    }
                )

    result = pd.DataFrame(cached_rows + scored_rows, columns=CACHE_COLUMNS)
    return result


def score_sentence_table(
    sentences_df: pd.DataFrame,
    cache_path=SENTIMENT_CACHE_PATH,
    only_relevant: bool = True,
) -> pd.DataFrame:
    """Score the unique texts in `sentences_df` and merge pos/neg/neu back on.

    With only_relevant=True (the default) only needs_score() rows reach FinBERT, which
    gives identical article features for ~60% fewer forward passes. Rows outside the
    mask get NaN, never 0.

    save_cache() runs once after all batches, not per batch, since parquet writes are
    full-file rewrites; an interrupted run therefore loses that run's progress. The
    save merges with the cache loaded at call start.
    """
    cache_df = load_cache(cache_path)

    n_total = len(sentences_df)
    if only_relevant and n_total > 0:
        mask = needs_score(sentences_df)
        texts = sentences_df.loc[mask, "text"].tolist()
        n_needed = int(mask.sum())
    else:
        mask = None
        texts = sentences_df["text"].tolist()
        n_needed = n_total

    n_unique = len({hash_text(t) for t in texts})
    pct_skipped = (100.0 * (n_total - n_needed) / n_total) if n_total else 0.0
    logger.info(
        f"score_sentence_table: {n_total} rows, {n_needed} need scores "
        f"(only_relevant={only_relevant}), {n_unique} unique texts to consider, "
        f"{pct_skipped:.1f}% of rows skipped"
    )

    scored = score_sentences(texts, cache_df=cache_df)

    # Merge, never replace. save_cache() dedupes on text_hash keep="last", and
    # `scored` is concatenated last, so freshly scored values win over any
    # stale entry for the same hash.
    save_cache(pd.concat([cache_df, scored], ignore_index=True), path=cache_path)

    sentences_df = sentences_df.copy()
    sentences_df["text_hash"] = sentences_df["text"].map(hash_text)
    if mask is not None:
        # Unscored rows get a null key so the left-merge leaves pos/neg/neu NaN
        # even when an identical text was scored for some other row.
        sentences_df.loc[~mask, "text_hash"] = None
    merged = sentences_df.merge(scored, on="text_hash", how="left")
    merged = merged.drop(columns=["text_hash"])
    for col in ("pos", "neg", "neu"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def score_headlines(headlines_df: pd.DataFrame, cache_path=SENTIMENT_CACHE_PATH) -> pd.DataFrame:
    """Score each headline standalone, one row per input row.

    One extra forward pass per article, since headlines carry disproportionate weight.
    Reuses score_sentences() and the sentence cache.

    The save MERGES, like score_sentence_table(). This function is the one that
    exposed the bug: a corpus run calls it straight after score_sentence_table(), so
    replacing the file wiped the sentence entries just written.
    """
    cache_df = load_cache(cache_path)
    headlines = headlines_df["headline"].tolist()

    scored = score_sentences(headlines, cache_df=cache_df)
    # Merge, never replace. save_cache() dedupes on text_hash keep="last", and
    # `scored` is concatenated last, so freshly scored values win over any
    # stale entry for the same hash.
    save_cache(pd.concat([cache_df, scored], ignore_index=True), path=cache_path)

    out = headlines_df.copy()
    out["text_hash"] = out["headline"].map(hash_text)
    merged = out.merge(scored, on="text_hash", how="left")
    return merged[["article_id", "pos", "neg", "neu"]]


def aggregate_article_features(
    sentences_df: pd.DataFrame, headline_scores: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Article-level aggregates from an already-scored sentence table.

    Never scores: `sentences_df` must already carry pos/neg/neu.

    Aggregates run over non-boilerplate sentences only; article_length still sums the
    full group, since it measures the article as published. maxmag stores the three
    raw probabilities of the largest |pos - neg| target sentence rather than one
    signed scalar.

    absa_* and fus_* are soft feature-detects: absent ABSA columns make both families
    NaN with one warning each, and the pipeline runs with ABSA off.

    Without `headline_scores` the sent_headline_* columns are NaN and one warning is
    logged.

    Empty sentence sets give NaN, never 0.
    """
    if len(sentences_df) == 0:
        return pd.DataFrame(
            columns=[
                "article_id",
                "sent_entity_pos",
                "sent_entity_neg",
                "sent_entity_neu",
                "sent_entity_maxmag_pos",
                "sent_entity_maxmag_neg",
                "sent_entity_maxmag_neu",
                "sent_entity_lead_pos",
                "sent_entity_lead_neg",
                "sent_entity_lead_neu",
                "sent_ceo_pos",
                "sent_ceo_neg",
                "sent_ceo_neu",
                "n_entity_sents",
                "n_boilerplate_sents",
                "n_ceo_sents",
                "n_total_sents",
                "entity_share",
                "article_length",
                "sent_headline_pos",
                "sent_headline_neg",
                "sent_headline_neu",
                *ABSA_FEATURE_COLUMNS,
                *FUSION_FEATURE_COLUMNS,
            ]
        )

    # Older sentence tables (and the single-article analyze() path before
    # boilerplate flagging existed) may not carry these columns; default them
    # to all-False once, up front, rather than guarding at every use site.
    sentences_df = sentences_df.copy().reset_index(drop=True)
    for col in ("is_boilerplate",):
        if col not in sentences_df.columns:
            sentences_df[col] = False
        else:
            sentences_df[col] = sentences_df[col].fillna(False).astype(bool)

    consumed = needs_score(sentences_df)

    has_absa = all(c in sentences_df.columns for c in ABSA_SCORE_COLUMNS)
    if not has_absa:
        logger.warning(
            "aggregate_article_features: no ABSA score columns "
            f"({', '.join(ABSA_SCORE_COLUMNS)}) on the sentence table; "
            f"the {len(ABSA_FEATURE_COLUMNS)} absa_* features will be all-NaN"
        )

    # Fusion needs both scorers, so it cannot run where ABSA could not.
    has_fusion = has_absa
    if not has_fusion:
        logger.warning(
            "aggregate_article_features: no ABSA score columns on the "
            f"sentence table; the {len(FUSION_FEATURE_COLUMNS)} fus_* "
            "features will be all-NaN"
        )
        fusion_features = pd.DataFrame(
            {"article_id": sentences_df["article_id"].unique()}
        ).assign(**{c: float("nan") for c in FUSION_FEATURE_COLUMNS})
    else:
        fusion_features = fusion.aggregate_fusion_features(sentences_df)

    def _absa_mean(frame) -> tuple[float, float, float]:
        """Mean absa_pos/neg/neu over `frame`, or NaN on an empty selection.

        Same no-signal-vs-measured-neutral rule as every FinBERT aggregate
        above: an empty bucket is NaN, never 0.
        """
        if not has_absa or len(frame) == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            frame["absa_pos"].mean(),
            frame["absa_neg"].mean(),
            frame["absa_neu"].mean(),
        )

    rows = []
    for article_id, group in sentences_df.groupby("article_id", sort=False):
        # n_total_sents / entity_share describe the non-boilerplate body; this
        # is the same "~is_boilerplate" term needs_score() applies.
        body = group[~group["is_boilerplate"]]
        grp_consumed = consumed.loc[group.index]

        # needs_score bucket 1 (mentions_target): sent_entity_*,
        # sent_entity_maxmag_*, sent_entity_lead_*.
        target = group[grp_consumed & group["mentions_target"]]
        lead_target = target[target["sent_idx"] < LEAD_SENTENCE_WINDOW]
        # needs_score bucket 3 (mentions_ceo): sent_ceo_*, narrowed here to
        # CEO-only sentences -- a subset of the mentions_ceo rows needs_score()
        # keeps, so it is covered by the predicate.
        ceo_only = group[grp_consumed & group["mentions_ceo"] & ~group["mentions_target"]]

        n_entity = len(target)
        n_total = len(body)
        n_boilerplate = int(group["is_boilerplate"].sum())
        n_ceo = len(ceo_only)

        if n_entity > 0:
            sent_entity_pos = target["pos"].mean()
            sent_entity_neg = target["neg"].mean()
            sent_entity_neu = target["neu"].mean()

            magnitude = (target["pos"] - target["neg"]).abs()
            maxmag_row = target.loc[magnitude.idxmax()]
            maxmag_pos = maxmag_row["pos"]
            maxmag_neg = maxmag_row["neg"]
            maxmag_neu = maxmag_row["neu"]
        else:
            sent_entity_pos = sent_entity_neg = sent_entity_neu = float("nan")
            maxmag_pos = maxmag_neg = maxmag_neu = float("nan")

        if len(lead_target) > 0:
            lead_pos = lead_target["pos"].mean()
            lead_neg = lead_target["neg"].mean()
            lead_neu = lead_target["neu"].mean()
        else:
            lead_pos = lead_neg = lead_neu = float("nan")

        if n_ceo > 0:
            ceo_pos = ceo_only["pos"].mean()
            ceo_neg = ceo_only["neg"].mean()
            ceo_neu = ceo_only["neu"].mean()
        else:
            ceo_pos = ceo_neg = ceo_neu = float("nan")

        absa_entity = _absa_mean(target)
        absa_ceo = _absa_mean(ceo_only)

        rows.append(
            {
                "article_id": article_id,
                "sent_entity_pos": sent_entity_pos,
                "sent_entity_neg": sent_entity_neg,
                "sent_entity_neu": sent_entity_neu,
                "sent_entity_maxmag_pos": maxmag_pos,
                "sent_entity_maxmag_neg": maxmag_neg,
                "sent_entity_maxmag_neu": maxmag_neu,
                "sent_entity_lead_pos": lead_pos,
                "sent_entity_lead_neg": lead_neg,
                "sent_entity_lead_neu": lead_neu,
                "sent_ceo_pos": ceo_pos,
                "sent_ceo_neg": ceo_neg,
                "sent_ceo_neu": ceo_neu,
                "n_entity_sents": n_entity,
                "n_boilerplate_sents": n_boilerplate,
                "n_ceo_sents": n_ceo,
                "n_total_sents": n_total,
                "entity_share": (n_entity / n_total) if n_total > 0 else 0,
                "article_length": group["char_len"].sum(),
                "absa_entity_pos": absa_entity[0],
                "absa_entity_neg": absa_entity[1],
                "absa_entity_neu": absa_entity[2],
                "absa_ceo_pos": absa_ceo[0],
                "absa_ceo_neg": absa_ceo[1],
                "absa_ceo_neu": absa_ceo[2],
            }
        )

    result = pd.DataFrame(rows)

    # fus_*: left-merge so every article_id in `result` keeps its row even if
    # fusion_features (built from the same sentences_df) somehow disagrees on
    # membership; in practice the two are built from the same article_id set.
    result = result.merge(fusion_features, on="article_id", how="left")

    if headline_scores is not None:
        headline_cols = headline_scores[["article_id", "pos", "neg", "neu"]].rename(
            columns={
                "pos": "sent_headline_pos",
                "neg": "sent_headline_neg",
                "neu": "sent_headline_neu",
            }
        )
        result = result.merge(headline_cols, on="article_id", how="left")
    else:
        logger.warning(
            "aggregate_article_features: no headline_scores provided; "
            "sent_headline_pos/neg/neu will be all-NaN"
        )
        result["sent_headline_pos"] = float("nan")
        result["sent_headline_neg"] = float("nan")
        result["sent_headline_neu"] = float("nan")

    return result


def analyze(
    article_text: str,
    ticker: str,
    headline: str | None = None,
    cache_path=SENTIMENT_CACHE_PATH,
) -> dict:
    """Single-article entry point for the live demo.

    Runs the same path as the batch pipeline (split, tag, score, aggregate), so there
    is no second implementation to drift. Returns one article's features as a dict.

    Unlike the batch functions this persists newly-scored text before returning.
    `cache_path` exists so tests can point somewhere disposable.
    """
    synthetic_article_id = "__analyze__"

    sentence_lists = entity_filter.split_sentences([article_text])
    sentences = sentence_lists[0] if sentence_lists else []

    sentences_df = entity_filter.tag_sentences(synthetic_article_id, sentences, ticker)

    cache_df = load_cache(cache_path)
    new_cache_frames = []

    if len(sentences_df) > 0:
        mask = needs_score(sentences_df)
        scored = score_sentences(sentences_df.loc[mask, "text"].tolist(), cache_df=cache_df)
        new_cache_frames.append(scored)
        sentences_df = sentences_df.copy()
        sentences_df["text_hash"] = sentences_df["text"].map(hash_text)
        sentences_df.loc[~mask, "text_hash"] = None
        sentences_df = sentences_df.merge(scored, on="text_hash", how="left").drop(
            columns=["text_hash"]
        )
        for col in ("pos", "neg", "neu"):
            sentences_df[col] = pd.to_numeric(sentences_df[col], errors="coerce")
    else:
        for col in ("pos", "neg", "neu"):
            sentences_df[col] = pd.Series(dtype=float)

    headline_scores = None
    if headline is not None:
        headline_scored = score_sentences([headline], cache_df=cache_df)
        new_cache_frames.append(headline_scored)
        headline_hash = hash_text(headline)
        match = headline_scored[headline_scored["text_hash"] == headline_hash]
        headline_scores = pd.DataFrame(
            {
                "article_id": [synthetic_article_id],
                "pos": [match["pos"].iloc[0]] if len(match) else [float("nan")],
                "neg": [match["neg"].iloc[0]] if len(match) else [float("nan")],
                "neu": [match["neu"].iloc[0]] if len(match) else [float("nan")],
            }
        )

    if new_cache_frames:
        combined_new = pd.concat(new_cache_frames, ignore_index=True)
        merged_cache = pd.concat([cache_df, combined_new], ignore_index=True)
        save_cache(merged_cache, path=cache_path)

    features_df = aggregate_article_features(sentences_df, headline_scores=headline_scores)

    if len(features_df) == 0:
        # No sentences at all (e.g. empty article text) -- build an all-NaN /
        # zero-count row so the return shape is still consistent.
        empty = aggregate_article_features(
            pd.DataFrame(columns=[*entity_filter.SENTENCE_COLUMNS, "pos", "neg", "neu"])
        )
        row = {c: float("nan") for c in empty.columns}
        row["article_id"] = synthetic_article_id
        row["n_entity_sents"] = 0
        row["n_boilerplate_sents"] = 0
        row["n_ceo_sents"] = 0
        row["n_total_sents"] = 0
        row["entity_share"] = 0
        row["article_length"] = 0
        return row

    return features_df.iloc[0].to_dict()


# The model-facing feature table. The wide table is for diagnosis and carries the
# raw triples; the model reads one score per population. See notebooks/text/2.3.
SHAPE_FEATURE_COLUMNS = [
    "n_total_sents",
    "n_entity_sents",
    "n_ceo_sents",
    "n_boilerplate_sents",
    "entity_share",
    "article_length",
]

# Derived from two fusion columns each. Explicit rather than left to the caller,
# for the reason fus_*_spread is: tree models split on single features and cannot
# form a difference between two of them.
DIVERGENCE_FEATURE_COLUMNS = ["fus_headline_gap", "fus_lead_gap"]

MODEL_FEATURE_COLUMNS = [
    *SHAPE_FEATURE_COLUMNS,
    *FUSION_FEATURE_COLUMNS,   # 6, all conf_graft_floor
    "fus_ceo_mean",
    "fus_headline",
    *fusion.EXTRA_FUSION_COLUMNS,
    *DIVERGENCE_FEATURE_COLUMNS,
]


# One line per model feature, keyed exactly as MODEL_FEATURE_COLUMNS. Used to
# emit the data dictionary that ships beside the parquet, so the description and
# the column cannot drift apart.
FEATURE_DESCRIPTIONS = {
    "n_total_sents": "Sentences in the article after splitting, excluding scraper residue.",
    "n_entity_sents": "Sentences tagged as being about the target, non-boilerplate.",
    "n_ceo_sents": "Sentences mentioning the CEO but not the target itself.",
    "n_boilerplate_sents": "Sentences whose exact text repeats across five or more articles.",
    "entity_share": "n_entity_sents over the article's non-boilerplate sentence count.",
    "article_length": "Characters in the article as published, boilerplate included.",
    "fus_conf_graft_floor_mean": "Mean fused score over the target sentences.",
    "fus_conf_graft_floor_median": "Median of the same population. Disagrees on sign with the mean on about a fifth of articles, so it detects skew.",
    "fus_conf_graft_floor_lead": "Mean over target sentences in the opening window only. 0.0, not NaN, when the window holds none.",
    "fus_conf_graft_floor_top3_pos": "Mean of the three highest sentence scores. Reads the tail rather than the centre.",
    "fus_conf_graft_floor_top3_neg": "Mean of the three lowest sentence scores.",
    "fus_conf_graft_floor_spread": "top3_pos minus top3_neg. Separates a contested article from a quiet one, which the mean cannot.",
    "fus_ceo_mean": "Mean fused score over CEO-only sentences. NaN where the article has none.",
    "fus_headline": "The headline through both scorers, grafted. One string, one number.",
    "fus_maxmag": "Signed score of the single loudest target sentence, chosen by largest absolute fused score.",
    "fus_trusted_mean": "Mean fused score over the surface and coref_span channels only, excluding the weakest-measured channel.",
    "fus_scorer_gap": "Mean absolute difference between the two scorers over the target population. High means the article was contested between them.",
    "fus_headline_gap": "fus_headline minus the body mean. How far the headline leads or lags the article.",
    "fus_lead_gap": "The lead score minus the body mean. How far the opening leads or lags the rest.",
}

def headline_names_target(headlines_df: pd.DataFrame, ticker: str) -> pd.Series:
    """Boolean mask: does each headline name the target explicitly?

    Uses entity_filter's compiled "names" pattern rather than a local regex, so a
    headline matches by the same rules a sentence does.

    Only the "names" tier counts; the person tier does not, matching the sentence-level
    rule that mentions_ceo never sets mentions_target.
    """
    pattern = entity_filter._build_ticker_patterns(ticker)["names"]
    text = headlines_df["headline"].fillna("").astype(str)
    if pattern is None:
        return pd.Series(False, index=headlines_df.index)
    return text.str.contains(pattern)


def build_model_features(
    sentences_df: pd.DataFrame,
    headline_finbert: pd.DataFrame,
    headline_absa: pd.DataFrame,
    headlines_df: pd.DataFrame | None = None,
    ticker: str = PRIMARY_TICKER,
) -> pd.DataFrame:
    """The lean, model-facing table: article_id + MODEL_FEATURE_COLUMNS.

    Composed from the same functions the wide table uses, so the two cannot disagree.
    `headline_finbert` and `headline_absa` are both required, since a fused headline
    needs both scorers on the same string.

    Every feature is NaN, never 0, where its population is empty.
    """
    wide = aggregate_article_features(sentences_df, headline_scores=headline_finbert)
    ceo = fusion.aggregate_ceo_fusion_features(sentences_df)
    head = fusion.headline_fusion_feature(headline_finbert, headline_absa)
    extra = fusion.aggregate_extra_fusion_features(sentences_df)

    out = (
        wide[["article_id", *SHAPE_FEATURE_COLUMNS, *FUSION_FEATURE_COLUMNS]]
        .merge(ceo, on="article_id", how="left")
        .merge(head, on="article_id", how="left")
        .merge(extra, on="article_id", how="left")
    )
    out = out.reindex(columns=["article_id", *MODEL_FEATURE_COLUMNS])

    # Body OR headline: roughly a quarter of the articles with no target sentence
    # in the body have an empty body, so filtering on the body alone would drop
    # real articles because extraction failed.
    if headlines_df is not None:
        head_hit = headline_names_target(headlines_df, ticker)
        relevant_ids = set(headlines_df.loc[head_hit, "article_id"])
        keep = (out["n_entity_sents"] > 0) | out["article_id"].isin(relevant_ids)
        out = out[keep].reset_index(drop=True)

        # Headline-only articles: no body sentiment, and 0.0 on a signed score
        # reads as exactly that. Decodable, since every filled row has
        # n_entity_sents == 0 and no other row does.
        filled = [*FUSION_FEATURE_COLUMNS, "fus_ceo_mean", *fusion.EXTRA_FUSION_COLUMNS]
        body_absent = out["n_entity_sents"] == 0
        out.loc[body_absent, filled] = out.loc[body_absent, filled].fillna(0.0)

    # Derived last, so they read the filled body mean rather than the NaN it
    # replaced. On a headline-only article the body mean is 0.0, so the headline
    # gap is the headline score itself, which is the honest reading.
    body_mean = out["fus_conf_graft_floor_mean"]
    out["fus_headline_gap"] = out["fus_headline"] - body_mean
    out["fus_lead_gap"] = out["fus_conf_graft_floor_lead"] - body_mean
    return out[["article_id", *MODEL_FEATURE_COLUMNS]]
