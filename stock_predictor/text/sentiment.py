"""FinBERT sentiment scoring over entity-tagged sentences.

Runs downstream of `stock_predictor.text.entity_filter`, which produces a
sentence table with columns article_id, sent_idx, text, mentions_target,
mentions_ceo, resolved_by_anaphora,
is_boilerplate, char_len.

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
  * Selective scoring: needs_score() is the single source of truth for which
    sentences aggregate_article_features() actually reads, and both the
    scorer and the aggregator route through it so they cannot drift. On the
    real corpus this cuts FinBERT forward passes by ~60% with byte-identical
    article features.
  * Cache persistence: score_sentence_table() and score_headlines() save the
    on-disk cache once at the end of their call (not per internal batch) --
    see score_sentence_table()'s docstring for the reasoning and the
    interruption-safety tradeoff this implies. Every save is a MERGE with the
    cache that was loaded at the start of the call (concat then dedupe),
    never a replacement. This matters because score_sentences() returns rows
    only for the hashes in the CURRENT call's input: saving that frame alone
    silently truncates the cache to just those hashes. Notebook 2.1 hit
    exactly this -- a full corpus run wrote ~56k sentence entries via
    score_sentence_table() and then score_headlines() overwrote the file with
    its ~2k headline entries, so the next run got ~2 cache hits out of ~56k
    unique sentences and paid for a full FinBERT pass again.
  * FinBERT label order is verified against model.config.id2label at load
    time rather than assumed -- see _build_label_index_map().
  * All three raw probabilities (pos, neg, neu) are stored everywhere; no
    single collapsed score is persisted anywhere in this module -- WITH ONE
    NAMED EXCEPTION. The fus_conf_graft_* / fus_conf_graft_soft_* /
    fus_conf_graft_floor_* columns
    (see FUSION_FEATURE_COLUMNS and the FUSION block in
    aggregate_article_features()) ARE signed scalars, not pos/neg/neu
    probability triples: they come from stock_predictor.text.fusion, which by
    design collapses FinBERT and ABSA into one graft-fused number per
    sentence (see fusion.py's module docstring for why). This is recorded
    here honestly rather than silently violated. The invariant is preserved
    IN SUBSTANCE, though: the raw pos/neg/neu and absa_pos/absa_neg/absa_neu
    columns those grafts are built from are all still stored on the sentence
    table, so every fus_* value remains exactly recomputable from what is
    persisted -- nothing is lost, a derived convenience is added on top.
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
from stock_predictor.text import entity_filter, fusion

CACHE_COLUMNS = ["text_hash", "pos", "neg", "neu"]

# Score columns an aspect-based pass (stock_predictor.text.absa) attaches to the
# sentence table, and the article-level columns aggregate_article_features()
# derives from them. These sit ALONGSIDE the FinBERT sent_* features: no
# existing column's name, semantics or value changes when ABSA is present, and
# every one of these is NaN when it is absent. See the ABSA block in
# aggregate_article_features().
ABSA_SCORE_COLUMNS = ["absa_pos", "absa_neg", "absa_neu"]
ABSA_FEATURE_COLUMNS = [
    "absa_entity_pos",
    "absa_entity_neg",
    "absa_entity_neu",
    "absa_ceo_pos",
    "absa_ceo_neg",
    "absa_ceo_neu",
]

# Article-level columns fusion.aggregate_fusion_features() derives from the
# conf_graft / conf_graft_soft / conf_graft_floor fusion variants (the last
# is the shipped scoring, see fusion.CONF_FLOOR) -- see the FUSION block in
# aggregate_article_features() below and fusion.aggregate_fusion_features()'s
# own docstring for the full reasoning (population, the six aggregations,
# why top3 exists, why K stays at 3, and the measured cases behind median and
# spread). Like ABSA_FEATURE_COLUMNS, this sits ALONGSIDE the
# existing sent_*/absa_* columns: nothing here changes any existing column's
# name, semantics or value, and every one of these is NaN when fusion cannot
# be computed (e.g. the ABSA score columns fusion needs are absent).
FUSION_FEATURE_COLUMNS = [
    f"fus_{variant}_{agg}"
    for variant in fusion.AGGREGATED_VARIANTS
    for agg in fusion.FUSION_AGGREGATIONS
]

# Sentence flags that put a sentence into a bucket some aggregate reads.
# See needs_score() -- this list is the enumeration behind that predicate.
CONSUMED_MENTION_COLUMNS = ["mentions_target", "mentions_ceo"]


def needs_score(sentences_df: pd.DataFrame) -> pd.Series:
    """Boolean mask of sentences whose FinBERT scores are actually consumed by
    aggregate_article_features().

    Definition:
        (mentions_target | mentions_ceo) & ~is_boilerplate

    THIS PREDICATE IS THE SINGLE SOURCE OF TRUTH shared by the scorer
    (score_sentence_table(), analyze()) and the aggregator
    (aggregate_article_features()), so the two cannot drift. Every
    score-derived column in aggregate_article_features() reads from exactly
    three sentence sets, all taken from the article's NON-boilerplate body:

        mentions_target -> sent_entity_*, sent_entity_maxmag_*, sent_entity_lead_*
        mentions_ceo    -> sent_ceo_* (CEO-only: mentions_ceo & ~mentions_target,
                           a subset of the mentions_ceo rows kept here)

    The mentions_other bucket was removed with other-company detection: nothing
    reads those rows any more, so scoring them was pure waste.

    A sentence in none of those buckets, and every boilerplate sentence, is
    never read by any feature: the remaining columns (n_total_sents,
    entity_share, article_length, and the n_*_sents counts) derive from row
    counts and char_len, not from scores. Scoring those rows is pure waste --
    on the real corpus only ~31% of sentences pass this predicate.

    IF A FUTURE FEATURE READS A NEW SENTENCE BUCKET, IT MUST BE ADDED HERE, or
    that feature silently gets NaN for every article. The gate test
    test_filtered_and_full_scoring_produce_identical_aggregates exists to make
    that failure loud.

    Ticker-agnostic by construction: it keys off the mentions_* flags, which
    entity_filter computes relative to whatever ticker was passed, so nothing
    here is target-specific.

    A missing is_boilerplate column defaults to all-False (nothing excluded on
    that basis), consistent with how aggregate_article_features() already
    handles it. The three mentions_* columns are required.

    Returns a bool Series aligned to sentences_df.index (empty frame -> empty
    bool Series).
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


def score_sentence_table(
    sentences_df: pd.DataFrame,
    cache_path=SENTIMENT_CACHE_PATH,
    only_relevant: bool = True,
) -> pd.DataFrame:
    """Score the unique sentence texts in `sentences_df` via score_sentences()
    with on-disk caching, then return sentences_df with pos/neg/neu merged on
    (deduped) text -> text_hash.

    Selective scoring (only_relevant=True, the DEFAULT): only rows where
    needs_score() is True are sent to FinBERT -- i.e. sentences in one of the
    three buckets aggregate_article_features() actually reads
    (mentions_target / mentions_ceo), excluding boilerplate.
    needs_score() is the single source of truth shared with the aggregator, so
    a filtered run and a full run produce byte-identical article features while
    the filtered run does ~60% fewer forward passes on the real corpus.

    Rows outside the mask get NaN for pos/neg/neu, never 0. This is the same
    no-signal-vs-measured-neutral rule the rest of the module follows: 0.0 is a
    real probability FinBERT could have returned, so writing it would silently
    conflate "never scored" with "scored and confidently not positive". No
    aggregate reads these rows, so NaN is safe -- and if some future code does
    read them, NaN propagates loudly instead of quietly poisoning a mean.

    Escape hatch: pass only_relevant=False to score every row (the old
    behavior), e.g. for exploratory analysis over the whole sentence table. Its
    marginal cost is low, because the on-disk cache is keyed by text hash and
    is ticker-agnostic: any score ever paid for is reused by later runs and by
    runs for other tickers.

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

    The save MERGES with the cache loaded at the start of the call rather than
    replacing it: score_sentences() returns rows only for this call's texts,
    so persisting that frame on its own would drop every previously cached
    entry not mentioned in `sentences_df` (see the module docstring's cache
    persistence note for the notebook 2.1 failure this caused).
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
    """Score each headline standalone (one extra forward pass per article --
    headlines carry disproportionate weight per spec). Reuses
    score_sentences() and the same on-disk cache as sentence scoring.

    Returns article_id, pos, neg, neu (one row per input row).

    Like score_sentence_table(), the save MERGES with the cache loaded at the
    start of the call rather than replacing it. This function is the one that
    exposed the bug: a corpus run calls it right after score_sentence_table(),
    so replacing the file with only the headline rows wiped the tens of
    thousands of sentence entries just written (see the module docstring's
    cache persistence note).
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
    """Compute article-level sentiment aggregates from an ALREADY-SCORED
    sentence table (never re-scores here -- sentences_df must already carry
    pos/neg/neu columns, e.g. from score_sentence_table()).

    Design decisions:
      * Zero-target-sentence articles get NaN (not 0) for all
        target-sentence-derived means. 0 would look like "neutral
        sentiment" (a real, scored value), silently conflating "no
        signal" with "measured and neutral". Same reasoning applied to
        sent_entity_lead_* when its sentence set is empty.
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
      * Boilerplate exclusion: every sentence-derived aggregate is computed
        over the article's NON-boilerplate sentences only (see
        entity_filter.flag_boilerplate()). Disclosure notices and syndicated
        filler get scored by FinBERT like real content and, because they
        repeat across hundreds of articles, pull every article toward the
        same value. n_total_sents and entity_share therefore describe the
        non-boilerplate body; article_length deliberately still sums over the
        FULL group, since it measures the physical length of the article as
        published, not the length of its usable content. The three
        score-consuming sentence sets are selected through needs_score(),
        which is the shared contract with score_sentence_table() -- see that
        function's docstring.
      * The sent_entity_excl_comp_* / absa_entity_excl_comp_* family and the
        sent_other_mean_* / absa_other_mean_* family were REMOVED along with
        other-company detection. excl_comp existed only because FinBERT scores a
        whole sentence and could not tell whose sentiment it was, so deleting
        comparative sentences was the only available defence; ABSA scores toward
        the aspect and reads them correctly (notebook 2.5 §1.2 already declined
        to build a fusion twin for exactly this reason). Both families depended
        on a registry of rival companies that no longer exists.

      * sent_ceo_*: mean over CEO-only sentences (mentions_ceo and NOT
        mentions_target). The 2.0 ablation showed folding CEO mentions into
        the target mean shifts the mean very little on average but touches
        31.5% of articles; exposing it as its own feature lets the model
        decide how much CEO talk is worth instead of us deciding here.
      * Columns whose sentence set is empty are NaN, never 0 — the same
        no-signal-vs-measured-neutral rule used throughout this module.
      * absa_*: aspect-aware twins of four of the sent_* means
        (absa_entity_*, absa_ceo_*), computed from the absa_pos/neg/neu columns
        stock_predictor.text.absa attaches to the sentence table. They read
        EXACTLY the same four sentence selections as their FinBERT
        counterparts, so a difference between a pair is a difference between
        the two MODELS and never between two populations. No existing sent_*
        column changes in name, semantics or value because of them. If the
        absa score columns are absent the whole family is NaN and one warning
        is logged — the pipeline works with ABSA off.
      * fus_*: article-level aggregates of the conf_graft / conf_graft_soft /
        conf_graft_floor fusion variants (stock_predictor.text.fusion),
        conf_graft_floor being the shipped scoring; computed by
        fusion.aggregate_fusion_features() and merged in here rather than
        inlined in this loop -- fusion.py owns the fusion logic, this module
        only wires it into the article feature table. Same soft feature-detect
        as ABSA: fusion needs both the FinBERT AND the ABSA score columns
        (score_variants() reads pos/neg/absa_pos/absa_neg/absa_neu), so if the
        ABSA score columns are absent the whole fus_* family is NaN and
        exactly ONE warning is logged -- the pipeline works with fusion off.
        See FUSION_FEATURE_COLUMNS and fusion.aggregate_fusion_features()'s
        own docstring for the population, the four aggregations, and why
        top3_pos/top3_neg exist.
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

    # The three score-consuming buckets below are derived THROUGH needs_score()
    # rather than re-deriving the boilerplate exclusion independently, so the
    # scorer and the aggregator cannot drift: score_sentence_table() scores
    # exactly the rows this mask marks, and every sentence whose pos/neg/neu is
    # read below is one of those rows. See needs_score()'s docstring.
    consumed = needs_score(sentences_df)

    # ABSA-parallel features. absa.score_sentence_table() attaches
    # absa_pos/neg/neu next to FinBERT's pos/neg/neu; when those columns are
    # absent (ABSA off, an older sentence table, or a backend that failed to
    # load) every absa_* aggregate is NaN and exactly ONE warning is logged.
    # The pipeline must work with ABSA off, so this is a soft feature-detect
    # rather than a required schema.
    has_absa = all(c in sentences_df.columns for c in ABSA_SCORE_COLUMNS)
    if not has_absa:
        logger.warning(
            "aggregate_article_features: no ABSA score columns "
            f"({', '.join(ABSA_SCORE_COLUMNS)}) on the sentence table; "
            f"the {len(ABSA_FEATURE_COLUMNS)} absa_* features will be all-NaN"
        )

    # fus_* features: fusion.score_variants() needs FinBERT's pos/neg AND
    # ABSA's absa_pos/absa_neg/absa_neu, so this reuses `has_absa` as the
    # gate rather than re-deriving a separate check -- fusion can never be
    # computed on a sentence table ABSA cannot be computed on either. Same
    # soft feature-detect and single-warning contract as ABSA above.
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

        # ABSA over the SAME four selections, so every absa_* column is the
        # aspect-aware twin of the sent_* column beside it and the pair is
        # directly comparable. Bucket membership is unchanged: it still comes
        # from needs_score() and the existing selections above.
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

    `cache_path` exists for the same reason score_sentence_table() has one:
    so a caller -- in practice the test suite -- can point the cache
    somewhere disposable. analyze() is the interactive entry point, so unlike
    the batch functions it writes on every call rather than once at the end;
    that makes it the one code path a careless test run would silently use to
    scribble synthetic sentences into data/interim/finbert_cache.parquet and
    bump its mtime. A test run must never mutate the project's real cache
    file, so this parameter is kept LAST (after the existing positional
    article_text/ticker/headline args) purely to avoid breaking any existing
    positional call site, and it defaults to SENTIMENT_CACHE_PATH so today's
    behavior is unchanged for every caller that doesn't pass it explicitly.
    """
    synthetic_article_id = "__analyze__"

    sentence_lists = entity_filter.split_sentences([article_text])
    sentences = sentence_lists[0] if sentence_lists else []

    sentences_df = entity_filter.tag_sentences(synthetic_article_id, sentences, ticker)

    cache_df = load_cache(cache_path)
    new_cache_frames = []

    if len(sentences_df) > 0:
        # Same needs_score() predicate as the batch path, for consistency and
        # cost: sentences in none of the consumed buckets are never read by
        # aggregate_article_features(), so scoring them is waste here too.
        # is_boilerplate is always False on the single-article path (it is
        # corpus-level), so this only skips the "neither bucket" sentences.
        # The returned feature dict is unchanged in keys and in values.
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
