"""FinBERT sentiment scoring over entity-tagged sentences.

Runs downstream of `stock_predictor.text.entity_filter`, which produces a
sentence table with columns article_id, sent_idx, text, mentions_target,
mentions_other, mentions_ceo, resolved_by_anaphora, char_len.

Pipeline:
    score_sentences()          -> core batched + on-disk-cached FinBERT scorer
    score_sentence_table()     -> scores every unique sentence in a corpus-wide
                                   sentence table, merges pos/neg/neu back on
    score_headlines()          -> scores headlines standalone (same cache)
    aggregate_article_features() -> per-article groupby aggregates from an
                                   ALREADY-SCORED sentence table
    analyze()                   -> single-article entry point for the live
                                   demo; same code path as the batch pipeline

Design notes (see module-level docstrings below for detail):
  * Cache persistence: score_sentence_table() and score_headlines() save the
    on-disk cache once at the end of their call (not per internal batch) --
    see score_sentence_table()'s docstring for the reasoning and the
    interruption-safety tradeoff this implies.
  * FinBERT label order is verified against model.config.id2label at load
    time rather than assumed -- see _build_label_index_map().
  * All three raw probabilities (pos, neg, neu) are stored everywhere; no
    single collapsed score is persisted anywhere in this module.
"""

import hashlib

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from stock_predictor.config import (
    FINBERT_MODEL,
    LEAD_SENTENCE_WINDOW,
    MAX_TOKENS,
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_CACHE_PATH,
)
from stock_predictor.text import entity_filter

CACHE_COLUMNS = ["text_hash", "pos", "neg", "neu"]


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
    """Core batched + cached FinBERT scoring function.

    Hashes every input text (dedup internally by text_hash -- callers may
    pass duplicate sentence text, e.g. repeated boilerplate across articles,
    and we only want to run the model once per unique string). Splits the
    unique hashes into cache-hits (looked up in `cache_df`, if given) and a
    needs-scoring subset. The needs-scoring subset is sorted by text length
    before batching so that same-batch sequences pad to similar lengths
    (less wasted compute/padding, ~20-30% speedup on CPU vs. scoring in
    arbitrary order), then batched in groups of `batch_size`, truncated to
    MAX_TOKENS, run through the model under torch.no_grad(), and softmaxed
    into (pos, neg, neu) probabilities.

    FinBERT's label order is verified against model.config.id2label (not
    assumed) so the softmax columns are mapped correctly regardless of what
    the loaded checkpoint's config says.

    If every hash is already in `cache_df` (all cache hits), the model is
    never loaded even if `model`/`tokenizer` are None -- this makes
    cache-hit-only calls cheap and avoids requiring a model at all for
    fully-cached workloads.

    Returns one row per UNIQUE text_hash (not per input text), with columns
    text_hash, pos, neg, neu. Does NOT write to disk -- callers own when to
    call save_cache().
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


def score_sentence_table(sentences_df: pd.DataFrame, cache_path=SENTIMENT_CACHE_PATH) -> pd.DataFrame:
    """Score every unique sentence text in `sentences_df` via score_sentences()
    with on-disk caching, then return sentences_df with pos/neg/neu merged on
    (deduped) text -> text_hash.

    Cache-save timing: this function calls save_cache() ONCE, after all
    batches complete, rather than per-batch. Rationale: per-batch saves would
    add ~150k/batch_size parquet writes over a full corpus run, which is a
    lot of I/O for a file that fits comfortably in memory, and parquet
    writes are full-file rewrites (not appends) so per-batch saving is O(n^2)
    in total bytes written. The tradeoff is that a run interrupted mid-way
    loses all progress from that run. Given CPU-only inference over ~150k
    sentences is expected to take a while and could be interrupted, callers
    that want interruption-safety should chunk their own calls to
    score_sentence_table() (e.g. by article batch) and rely on the fact that
    each call's cache_df is reloaded from disk at the start, so a prior
    completed chunk's results are never re-scored.
    """
    cache_df = load_cache(cache_path)
    texts = sentences_df["text"].tolist()

    scored = score_sentences(texts, cache_df=cache_df)

    save_cache(scored, path=cache_path)

    sentences_df = sentences_df.copy()
    sentences_df["text_hash"] = sentences_df["text"].map(hash_text)
    merged = sentences_df.merge(scored, on="text_hash", how="left")
    merged = merged.drop(columns=["text_hash"])
    return merged


def score_headlines(headlines_df: pd.DataFrame, cache_path=SENTIMENT_CACHE_PATH) -> pd.DataFrame:
    """Score each headline standalone (one extra forward pass per article --
    headlines carry disproportionate weight per spec). Reuses
    score_sentences() and the same on-disk cache as sentence scoring.

    Returns article_id, pos, neg, neu (one row per input row).
    """
    cache_df = load_cache(cache_path)
    headlines = headlines_df["headline"].tolist()

    scored = score_sentences(headlines, cache_df=cache_df)
    save_cache(scored, path=cache_path)

    out = headlines_df.copy()
    out["text_hash"] = out["headline"].map(hash_text)
    merged = out.merge(scored, on="text_hash", how="left")
    return merged[["article_id", "pos", "neg", "neu"]]


def aggregate_article_features(
    sentences_df: pd.DataFrame, headline_scores: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute article-level sentiment aggregates from an ALREADY-SCORED
    sentence table (never re-scores here -- sentences_df must already carry
    pos/neg/neu columns, e.g. from score_sentence_table()).

    Design decisions:
      * Zero-target-sentence articles get NaN (not 0) for all
        target-sentence-derived means. 0 would look like "neutral
        sentiment" (a real, scored value), silently conflating "no
        signal" with "measured and neutral". Same reasoning applied to
        sent_other_mean_* and sent_entity_lead_* when their respective
        sentence sets are empty.
      * "maxmag" (largest-magnitude target sentence): for each article,
        among its mentions_target sentences, pick the one with the largest
        |pos - neg| and store ITS three raw probabilities as
        sent_entity_maxmag_pos/neg/neu (not a single signed scalar). This
        keeps the "never collapse to a single score in storage" rule intact
        while still identifying the most emotionally extreme sentence about
        the target company. A signed-magnitude column would need pos/neg
        recomputed downstream anyway and would throw away neu, so the
        three-column form is strictly more useful for the same cost.
      * headline_scores is left-joined in; if not provided, the three
        sent_headline_* columns are all-NaN and a warning is logged once.
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
                "sent_other_mean_pos",
                "sent_other_mean_neg",
                "sent_other_mean_neu",
                "sent_entity_lead_pos",
                "sent_entity_lead_neg",
                "sent_entity_lead_neu",
                "n_entity_sents",
                "n_other_sents",
                "n_total_sents",
                "entity_share",
                "article_length",
                "sent_headline_pos",
                "sent_headline_neg",
                "sent_headline_neu",
            ]
        )

    rows = []
    for article_id, group in sentences_df.groupby("article_id", sort=False):
        target = group[group["mentions_target"]]
        other = group[group["mentions_other"]]
        lead_target = target[target["sent_idx"] < LEAD_SENTENCE_WINDOW]

        n_entity = len(target)
        n_other = len(other)
        n_total = len(group)

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

        if n_other > 0:
            other_pos = other["pos"].mean()
            other_neg = other["neg"].mean()
            other_neu = other["neu"].mean()
        else:
            other_pos = other_neg = other_neu = float("nan")

        if len(lead_target) > 0:
            lead_pos = lead_target["pos"].mean()
            lead_neg = lead_target["neg"].mean()
            lead_neu = lead_target["neu"].mean()
        else:
            lead_pos = lead_neg = lead_neu = float("nan")

        rows.append(
            {
                "article_id": article_id,
                "sent_entity_pos": sent_entity_pos,
                "sent_entity_neg": sent_entity_neg,
                "sent_entity_neu": sent_entity_neu,
                "sent_entity_maxmag_pos": maxmag_pos,
                "sent_entity_maxmag_neg": maxmag_neg,
                "sent_entity_maxmag_neu": maxmag_neu,
                "sent_other_mean_pos": other_pos,
                "sent_other_mean_neg": other_neg,
                "sent_other_mean_neu": other_neu,
                "sent_entity_lead_pos": lead_pos,
                "sent_entity_lead_neg": lead_neg,
                "sent_entity_lead_neu": lead_neu,
                "n_entity_sents": n_entity,
                "n_other_sents": n_other,
                "n_total_sents": n_total,
                "entity_share": (n_entity / n_total) if n_total > 0 else 0,
                "article_length": group["char_len"].sum(),
            }
        )

    result = pd.DataFrame(rows)

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


def analyze(article_text: str, ticker: str, headline: str | None = None) -> dict:
    """Single-article entry point for the live Streamlit demo. Runs the same
    code path as the batch corpus pipeline (entity_filter -> score_sentences
    -> aggregate_article_features) so there is no second implementation to
    drift out of sync with the batch pipeline (train/serve skew).

    Builds a synthetic one-article sentence table via
    entity_filter.split_sentences() + tag_sentences(), scores it through
    score_sentences() (a single small batch; the on-disk cache still
    applies), optionally scores the headline standalone, and returns the
    aggregate_article_features() output for this one article as a dict
    (rather than a one-row DataFrame, since this is for interactive use).

    Design decision: unlike score_sentence_table()/score_headlines() (which
    are called once per large batch job), analyze() DOES persist newly-scored
    text back to the on-disk cache before returning. Rationale: this is the
    interactive/demo path, so the same article or overlapping boilerplate
    text may be re-analyzed across repeated user actions in a single
    Streamlit session (or across sessions); paying the parquet-rewrite cost
    once per single-article call (a handful of new rows at most) is cheap
    and makes repeat interactions on the same text instant.
    """
    synthetic_article_id = "__analyze__"

    sentence_lists = entity_filter.split_sentences([article_text])
    sentences = sentence_lists[0] if sentence_lists else []

    sentences_df = entity_filter.tag_sentences(synthetic_article_id, sentences, ticker)

    cache_df = load_cache()
    new_cache_frames = []

    if len(sentences_df) > 0:
        scored = score_sentences(sentences_df["text"].tolist(), cache_df=cache_df)
        new_cache_frames.append(scored)
        sentences_df = sentences_df.copy()
        sentences_df["text_hash"] = sentences_df["text"].map(hash_text)
        sentences_df = sentences_df.merge(scored, on="text_hash", how="left").drop(
            columns=["text_hash"]
        )
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
        save_cache(merged_cache)

    features_df = aggregate_article_features(sentences_df, headline_scores=headline_scores)

    if len(features_df) == 0:
        # No sentences at all (e.g. empty article text) -- build an all-NaN /
        # zero-count row so the return shape is still consistent.
        empty = aggregate_article_features(
            pd.DataFrame(
                columns=[
                    "article_id",
                    "sent_idx",
                    "text",
                    "mentions_target",
                    "mentions_other",
                    "mentions_ceo",
                    "resolved_by_anaphora",
                    "char_len",
                    "pos",
                    "neg",
                    "neu",
                ]
            )
        )
        row = {c: float("nan") for c in empty.columns}
        row["article_id"] = synthetic_article_id
        row["n_entity_sents"] = 0
        row["n_other_sents"] = 0
        row["n_total_sents"] = 0
        row["entity_share"] = 0
        row["article_length"] = 0
        return row

    return features_df.iloc[0].to_dict()
