"""A hand-labelled gold set for entity-scoped sentiment: sample + blank sheet.

WHY THIS EXISTS. Every validation this project has produced so far is an
eyeball check: a handful of adjudicated articles in a notebook, a 10-15
sentence spot check in a docstring's "hand-inspection found..." aside. None of
that is an accuracy NUMBER. There is no answer to "is FinBERT-on-a-sentence or
ABSA-with-an-injected-aspect actually right more often, and on which kind of
sentence." A full evaluation would tie sentiment to forward returns, which is
slow, confounded by everything else that moves a stock, and overkill for the
question "did the model read this sentence correctly." The cheap intermediate
is a GOLD SET: a few hundred sentences, drawn so that every tagging MECHANISM
in entity_filter.py (an explicit mention, a comparative sentence, an anaphora
resolution, a coref resolution, a CEO-only mention, an other-company-only
mention) is represented in the sample, labelled by a human for sentiment
TOWARD THE TARGET COMPANY -- the exact question the pipeline is trying to
answer, not "is this sentence positive." Joining those labels back to the
model's own predictions is what finally produces a real accuracy figure,
comparable across FinBERT and ABSA and across strata.

WHAT THIS MODULE DOES NOT DO. It does not label anything. Labelling is the
human annotator's job, done against references/goldset-protocol.md, off a CSV
this module writes with every model output withheld. A machine-generated label
folded into "ground truth" would make the resulting accuracy number circular --
it would just re-measure whether the model agrees with itself. So the module's
job stops at building the SAMPLE (stratified across mechanisms, capped so no
one article dominates, reproducible under a fixed seed) and the two files that
come out of it: a blank annotation sheet for the human, and a predictions file
carrying the withheld model columns to be joined back in AFTER labelling.

Pipeline:
    assign_stratum()        -> one mutually-exclusive stratum label per row,
                                from the tagging-mechanism columns entity_filter
                                produces (see SENTENCE_COLUMNS in that module)
    stratified_sample()     -> reproducible, per-article-capped sample across
                                strata
    write_annotation_sheet()-> the blank CSV a human labels, model-output-free
    write_predictions()     -> the withheld model columns, joined back on
                                sample_id after labelling to score accuracy
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# The stratum vocabulary. NOTE: this tuple is in reporting order (the easy,
# most populous case first), NOT in the order assign_stratum() applies its
# rules -- that order is comparative, coref_resolved, anaphora_resolved,
# direct, ceo_only, other_only, and the reasoning for it lives in that
# function's docstring. stratified_sample() iterates this tuple only to decide
# which strata to draw, which is order-independent.
STRATA = (
    "direct",
    "comparative",
    "coref_resolved",
    "anaphora_resolved",
    "ceo_only",
    "other_only",
)

# Cap on how many sampled sentences may come from one article_id. Without this,
# a handful of unusually long articles (or a single article that happens to be
# the corpus's one big comparative-sentence cluster) could supply a large
# fraction of a stratum's sample, and the resulting accuracy number would
# really be measuring how well the model reads those two or three articles
# rather than the stratum in general.
MAX_PER_ARTICLE = 2

# Default seed. Fixed (not datetime-derived) so that re-running
# stratified_sample() on the same input frame reproduces the exact same
# sample_id set -- re-sampling must not orphan labels already collected for a
# prior draw.
DEFAULT_SEED = 20260816

# Column order of the human-facing annotation sheet. Deliberately short: see
# write_annotation_sheet()'s docstring for the anti-anchoring rule that keeps
# every model output off this list.
_ANNOTATION_COLUMNS = ["sample_id", "stratum", "absa_aspect", "text", "label", "confidence", "notes"]

# Withheld model-output columns write_predictions() may carry, if present on
# the input frame. "May" because an older sentence table predates ABSA/coref
# and simply won't have some of these -- see that function's docstring.
_PREDICTION_SCORE_COLUMNS = [
    "pos",
    "neg",
    "neu",
    "absa_pos",
    "absa_neg",
    "absa_neu",
    "absa_text",
]
_PREDICTION_FLAG_COLUMNS = [
    "mentions_target",
    "mentions_other",
    "mentions_ceo",
    "is_comparative",
    "resolved_by_coref",
    "resolved_by_anaphora",
    "other_source",
]


def assign_stratum(sentences_df: pd.DataFrame) -> pd.Series:
    """One stratum label per row (see STRATA), or "" for a row that belongs to
    no stratum or that must never be labelled at all.

    Strata are MUTUALLY EXCLUSIVE, assigned in this priority order:

      1. "comparative" -- is_comparative (an explicit target mention AND an
         explicit registry other-company mention in the same sentence).
         Highest priority because comparative sentences are the whole reason
         absa.py exists: "Build-A-Bear outperformed Tesla" is unambiguously
         positive as a sentence and unambiguously negative toward Tesla, and
         sentiment.aggregate_article_features() carries a whole parallel
         *_excl_comp_* feature family because of this exact failure mode. If
         accuracy is measured anywhere first, it should be measured here.
      2. "coref_resolved" -- resolved_by_coref. Neural coreference is the more
         expensive, more precision-sensitive of the two resolution mechanisms
         (coref.py), so it is checked before the cheaper heuristic.
      3. "anaphora_resolved" -- resolved_by_anaphora. entity_filter guarantees
         resolved_by_coref and resolved_by_anaphora are never both True on one
         row (see resolve_anaphora()'s docstring), so this and the previous
         rule never actually compete for the same row; the ordering is here
         for readability, not correctness.
      4. "direct" -- mentions_target, and not resolved by either mechanism
         above: the target was named explicitly in the sentence itself. This
         sits below the resolution strata on purpose -- a "direct" mention is
         the easy case models are expected to get right, so it is checked last
         among the target-mention rules to make sure a row that could ALSO
         count as comparative/coref/anaphora-resolved is filed under the more
         diagnostic bucket, not the easy one.
      5. "ceo_only" -- mentions_ceo and not mentions_target. The CEO-alias
         tier is deliberately never allowed to imply mentions_target (see
         entity_filter.tag_sentences()), so this stratum asks a genuinely
         different question: how does the model handle "Musk"-only sentences
         attributed to no company at all.
      6. "other_only" -- mentions_other and not mentions_target. The company
         the sentence names is not the target at all; this stratum's accuracy
         figure is really a check on the aspect-substitution/resolution
         machinery (absa._other_aspect(), _substitute_resolved()) rather than
         on sentiment scoring per se, but it shares the same annotation
         mechanics so it rides along here.

    A row that matches none of the above (mentions nothing the pipeline
    tracks) gets "". A row with is_boilerplate True gets "" REGARDLESS of what
    else it matches -- flag_boilerplate() exists precisely to mark sentences
    that are excluded from every downstream aggregate, and needs_score()
    (sentiment.py) never scores them; labelling one would produce a label for
    a sentence the pipeline itself never reads.

    Returns a string Series aligned to sentences_df.index.
    """
    if len(sentences_df) == 0:
        return pd.Series(dtype=object, index=sentences_df.index)

    def _bool_col(name: str) -> pd.Series:
        if name in sentences_df.columns:
            return sentences_df[name].fillna(False).astype(bool)
        return pd.Series(False, index=sentences_df.index)

    is_comparative = _bool_col("is_comparative")
    resolved_by_coref = _bool_col("resolved_by_coref")
    resolved_by_anaphora = _bool_col("resolved_by_anaphora")
    mentions_target = _bool_col("mentions_target")
    mentions_ceo = _bool_col("mentions_ceo")
    mentions_other = _bool_col("mentions_other")
    is_boilerplate = _bool_col("is_boilerplate")

    stratum = pd.Series("", index=sentences_df.index, dtype=object)

    # Priority order: later assignments only apply where stratum is still "",
    # so each row keeps the FIRST (highest-priority) rule it matches.
    unset = stratum == ""
    stratum = stratum.where(~(unset & is_comparative), "comparative")

    unset = stratum == ""
    stratum = stratum.where(~(unset & resolved_by_coref), "coref_resolved")

    unset = stratum == ""
    stratum = stratum.where(~(unset & resolved_by_anaphora), "anaphora_resolved")

    unset = stratum == ""
    direct = mentions_target & ~resolved_by_coref & ~resolved_by_anaphora
    stratum = stratum.where(~(unset & direct), "direct")

    unset = stratum == ""
    ceo_only = mentions_ceo & ~mentions_target
    stratum = stratum.where(~(unset & ceo_only), "ceo_only")

    unset = stratum == ""
    other_only = mentions_other & ~mentions_target
    stratum = stratum.where(~(unset & other_only), "other_only")

    # Boilerplate rows are excluded from scoring and must not be labelled,
    # regardless of what tagging mechanism they otherwise match.
    stratum = stratum.where(~is_boilerplate, "")

    return stratum


def stratified_sample(
    sentences_df: pd.DataFrame,
    n_per_stratum: int = 50,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """A reproducible, per-article-capped sample of up to `n_per_stratum` rows
    from each stratum in STRATA.

    Rows with an empty stratum (assign_stratum() == "") are excluded first --
    they are either untagged by any mechanism or boilerplate, neither of which
    should ever reach a human annotator.

    Sampling is WITHOUT replacement, per stratum, under a numpy Generator
    seeded with `seed`: the same input frame and the same seed always produce
    the same sample_id set, so a gold set can be regenerated (e.g. after new
    articles are added to sentences_df) and the previously-collected labels
    re-joined on sample_id rather than thrown away and re-collected.

    If a stratum has fewer eligible rows than `n_per_stratum`, ALL of its rows
    are taken and a warning is logged naming the stratum and the shortfall --
    silently returning a short sample would hide a real coverage gap (e.g. a
    corpus with too few coref-resolved sentences to say anything about that
    mechanism at all).

    MAX_PER_ARTICLE caps how many of a stratum's sampled rows may share one
    article_id (module constant, default 2). This is applied by sampling
    candidates in a random order and taking rows greedily subject to the
    per-article cap, rather than sampling first and capping after -- capping
    after would silently shrink the sample below n_per_stratum even when the
    stratum has plenty of eligible rows spread across many articles. The
    protection matters because a single long or heavily-syndicated article can
    otherwise supply a large share of a stratum, and the resulting accuracy
    number would measure how well the model reads that one article rather than
    the stratum in general.

    Adds two columns:
      * "stratum"   -- the stratum this row was drawn for (from assign_stratum()).
      * "sample_id" -- f"{stratum}-{article_id}-{sent_idx}", a STABLE,
        DETERMINISTIC id (never a random uuid) so that re-running this function
        against an updated sentences_df reproduces the same id for the same
        underlying sentence, letting previously-collected labels be re-joined
        by sample_id after a re-sample.

    Returns the sampled rows (original columns plus stratum and sample_id),
    sorted by sample_id.
    """
    if len(sentences_df) == 0:
        empty = sentences_df.copy()
        empty["stratum"] = pd.Series(dtype=object)
        empty["sample_id"] = pd.Series(dtype=object)
        return empty

    stratum = assign_stratum(sentences_df)
    eligible = sentences_df.loc[stratum != ""].copy()
    eligible["stratum"] = stratum.loc[eligible.index]

    rng = np.random.default_rng(seed)

    sampled_frames = []
    for stratum_name in STRATA:
        pool = eligible.loc[eligible["stratum"] == stratum_name]
        n_available = len(pool)
        if n_available == 0:
            logger.warning(
                f"stratified_sample: stratum '{stratum_name}' has no eligible "
                f"rows (wanted {n_per_stratum})"
            )
            continue
        if n_available < n_per_stratum:
            logger.warning(
                f"stratified_sample: stratum '{stratum_name}' has only "
                f"{n_available} eligible row(s), fewer than the requested "
                f"{n_per_stratum} (shortfall {n_per_stratum - n_available}); "
                "taking all of them"
            )

        # Shuffle candidate order deterministically, then take rows greedily
        # subject to MAX_PER_ARTICLE, so the per-article cap does not shrink
        # the sample below n_per_stratum when enough distinct articles exist.
        order = rng.permutation(n_available)
        pool_positional = pool.reset_index(drop=True)
        pool_article_ids = pool_positional["article_id"].tolist()

        picked_positions = []
        per_article_count: dict = {}
        for pos in order:
            if len(picked_positions) >= n_per_stratum:
                break
            article_id = pool_article_ids[pos]
            count = per_article_count.get(article_id, 0)
            if count >= MAX_PER_ARTICLE:
                continue
            per_article_count[article_id] = count + 1
            picked_positions.append(pos)

        if picked_positions:
            sampled_frames.append(pool_positional.iloc[picked_positions])

    if not sampled_frames:
        empty = sentences_df.copy()
        empty["stratum"] = pd.Series(dtype=object)
        empty["sample_id"] = pd.Series(dtype=object)
        return empty.iloc[0:0]

    sample = pd.concat(sampled_frames)
    sample["sample_id"] = (
        sample["stratum"].astype(str)
        + "-"
        + sample["article_id"].astype(str)
        + "-"
        + sample["sent_idx"].astype(str)
    )
    sample = sample.sort_values("sample_id").reset_index(drop=True)
    return sample


def write_annotation_sheet(sample_df: pd.DataFrame, path) -> None:
    """Write the blank CSV a human annotator labels.

    Columns, in this exact order: sample_id, stratum, absa_aspect, text,
    label, confidence, notes. label/confidence/notes are written EMPTY --
    filling them in is the annotator's job, following
    references/goldset-protocol.md.

    THE ANTI-ANCHORING RULE. This sheet must contain NO model output: no
    FinBERT pos/neg/neu, no absa_pos/absa_neg/absa_neu, no absa_text (which
    would reveal what the pipeline believed an anaphor resolved to -- seeing
    "Tesla also raised prices" when the real sentence read "The company also
    raised prices" tips the annotator off to the model's own resolution
    decision), and no mentions_* flags (which reveal which tagging mechanism
    produced the row, i.e. which answer the pipeline already committed to). An
    annotator shown a model's own answer tends to agree with it -- this is a
    well-documented anchoring effect, not a hypothetical -- and the resulting
    accuracy figure would then measure agreement with the model rather than
    correctness. Since the whole point of a gold set is an independent
    accuracy number, anchoring here would make the exercise circular.

    absa_aspect IS included, deliberately, as the one exception: the annotator
    is being asked "what is the sentiment toward THIS company", and
    absa_aspect names which company that is. That is the question being
    posed, not an answer to it -- omitting it would leave the annotator
    guessing which of several companies mentioned in the sentence they are
    supposed to judge.

    stratum is also included, so the annotator (per the protocol document) can
    be told the labelling rule varies by stratum -- e.g. comparative
    sentences are judged toward the aspect company only, and
    anaphora/coref-resolved sentences are labelled "unclear" when the
    referent cannot be recovered from the sentence alone (see the protocol
    doc). This is a minor, deliberately accepted anchoring risk: knowing the
    stratum could bias an annotator's expectation of what the "right" answer
    looks like. It is accepted in exchange for traceability -- without it,
    per-stratum accuracy could not be computed after the fact, which is the
    entire reason this module stratifies in the first place.

    Rows are SHUFFLED before writing (under DEFAULT_SEED, so the shuffle order
    is itself reproducible run over run for the same sample -- this function
    takes no seed argument on purpose, since the sheet's row order is a
    presentation detail rather than part of the sample's identity), so strata
    are interleaved rather than presented as contiguous blocks. Blocked
    presentation ("fifty comparative sentences in a row, then fifty
    anaphora-resolved ones") induces streak effects: an annotator settles into
    a rhythm calibrated to one stratum's typical difficulty and label
    distribution, and that rhythm carries over incorrectly into the next
    block. Interleaving keeps every judgement independent of the ones around
    it.

    Creates the parent directory of `path` if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = sample_df.copy()
    if "absa_aspect" not in out.columns:
        out["absa_aspect"] = ""
    out["absa_aspect"] = out["absa_aspect"].fillna("")
    out["label"] = ""
    out["confidence"] = ""
    out["notes"] = ""

    # Deterministic shuffle: same sample -> same on-disk row order every time
    # this is called, but strata interleaved rather than block-presented.
    rng = np.random.default_rng(DEFAULT_SEED)
    shuffled_positions = rng.permutation(len(out))
    out = out.iloc[shuffled_positions].reset_index(drop=True)

    sheet = out[_ANNOTATION_COLUMNS]
    sheet.to_csv(path, index=False)
    logger.info(f"write_annotation_sheet: wrote {len(sheet)} rows to {path}")


def write_predictions(sample_df: pd.DataFrame, path) -> None:
    """Write the withheld model columns to a SEPARATE csv, for joining back to
    the human labels on sample_id AFTER labelling to compute accuracy.

    Columns, when present on `sample_df`: sample_id, pos, neg, neu, absa_pos,
    absa_neg, absa_neu, absa_text, plus every mentions_*/resolved_by_*/
    other_source column present. Any of these that is missing from
    `sample_df` is skipped rather than raising -- an older sentence table may
    predate ABSA (no absa_* columns) or coref (no resolved_by_coref), and this
    module must still be usable against it; the gold set's value does not
    depend on every model being present, only on whatever IS present being
    withheld from the annotation sheet and recoverable afterward.

    Creates the parent directory of `path` if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["sample_id"] + [
        c for c in (_PREDICTION_SCORE_COLUMNS + _PREDICTION_FLAG_COLUMNS) if c in sample_df.columns
    ]
    predictions = sample_df[columns].copy()
    predictions.to_csv(path, index=False)
    logger.info(
        f"write_predictions: wrote {len(predictions)} rows with columns "
        f"{columns} to {path}"
    )
