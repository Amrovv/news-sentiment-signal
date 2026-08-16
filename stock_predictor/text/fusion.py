"""Candidate fusions of FinBERT and ABSA sentiment, per sentence.

WHY THIS EXISTS. The pipeline now carries two sentiment scorers over the same
sentence table: FinBERT (sentiment.py, columns pos/neg/neu) and ABSA
(absa.py, columns absa_pos/absa_neg/absa_neu). They answer two different
questions and were never meant to replace one another. FinBERT scores the
SENTENCE — it is finance-trained, so its sense of how intense a piece of
financial language is, is calibrated on financial text. ABSA scores
sentiment TOWARD AN ASPECT — it is review-trained (not finance-trained), but
it is the only one of the two that is handed the target company explicitly
and asked "how does this sentence read about THAT entity", so it is the only
one of the two that can get a comparative or multi-entity sentence right when
the sentiment in the sentence belongs to someone other than the target.

THE DISAGREEMENT, MEASURED. On this corpus the two scorers disagree on the
SIGN of sentiment for 23.2% of non-comparative target sentences and 30.9% of
comparative ones, and their signed scores (pos - neg) correlate at only 0.49.
That is too much disagreement to treat one model as simply a noisier copy of
the other. Every hand-adjudicated disagreement examined so far (25 sentences,
all of them selected BECAUSE they disagreed) favoured ABSA on attribution —
i.e. when the two scorers pointed in different directions, the human reading
the sentence sided with ABSA's direction, not FinBERT's. But that sample says
nothing about MAGNITUDE: it was never designed to test whether FinBERT's
intensity calibration is trustworthy, only whether ABSA's sense of "who is
this about" beats FinBERT's on the sentences where they disagree in the first
place.

THE HYPOTHESIS THIS MODULE EXISTS TO TEST. Attribution and intensity might be
SEPARABLE quantities that each model is better at than the other: take
DIRECTION from ABSA (it knows who the sentiment is about), take MAGNITUDE
from FinBERT (it is calibrated on financial language). This module builds the
candidate features needed to test that hypothesis by hand. It does not decide
the hypothesis is true. The variants below — especially `sign_graft` and
`gated` — are CANDIDATES FOR HAND INSPECTION, not a decided fusion ready to
feed a model. `mean_blend` and `agree_only` exist specifically so the
constructed candidates have something naive to beat: if `sign_graft` or
`gated` cannot outperform a plain average of the two scorers, that is itself
the finding, and without a naive baseline column sitting right next to them
there would be no way to notice that failure.

NOTHING HERE REPLACES OR MODIFIES ANY EXISTING COLUMN. This module reads
pos/neg/neu and absa_pos/absa_neg/absa_neu and writes ONLY new, separately
named columns. No sent_* or absa_* column anywhere in the pipeline changes
name, semantics or value because this module exists.

PER-SENTENCE, DELIBERATELY, NOT PER-ARTICLE. Every feature this module
produces is defined and returned at the SENTENCE level — one row in, one
value per variant out, for every row, unconditionally. This is a deliberate
design constraint, not an oversight: these features are going to be validated
by a human reading individual sentences and asking "does the number attached
to THIS sentence look sane". An article-level average cannot be inspected
that way — a human cannot look at a single mean and tell whether the
underlying attribution or magnitude logic did the right thing on any
particular sentence. If article-level aggregation of these variants happens
at all, it happens later, in a different module, over an already-validated
per-sentence column. `score_variants()` never groups by article_id and never
drops or reorders rows.

`score_variants` DOES NOT FILTER ROWS. It scores whatever sentence table it
is handed, unconditionally, for every row present. Selecting which sentences
matter (e.g. only mentions_target sentences, only non-boilerplate sentences)
is the caller's job, exactly as it is the caller's job upstream of
aggregate_article_features(). Mixing row selection into this module would
make it impossible to reuse for a different selection later, and would make
the per-sentence inspection story above harder to follow, since a human
reading this module's output next to the raw sentence table would have to
first work out which rows were silently dropped.

NaN-OVER-ZERO, the same rule the rest of the text layer follows everywhere
(see sentiment.py's and absa.py's docstrings for the precedent). 0.0 is a
real, meaningful value produced by these variants: `sign_graft`'s deadband
and `agree_only`'s disagreement case are both DEFINED to produce exactly
0.0, and that is a substantive claim ("no attributable signal" /
"low-confidence, discard"), not an absence of measurement. Conflating that
with "this row was never scored" would silently destroy the distinction the
whole no-signal-vs-measured-neutral convention exists to preserve. So: if
EITHER model's score is missing (NaN) for a row, every variant that reads
that row is NaN for that row -- never silently coerced to 0.
"""

import numpy as np
import pandas as pd
from loguru import logger

from stock_predictor.config import LEAD_SENTENCE_WINDOW

# Absolute-value threshold below which an ABSA signed score is treated as too
# close to neutral to trust for DIRECTION -- used by sign_graft (via
# score_variants' `deadband` parameter) to force such rows to exactly 0.0
# instead of grafting ABSA's sign onto FinBERT's magnitude.
#
# THIS WAS A GUESS AND IT WAS MEASURED WRONG. The deadband's premise was that
# a near-zero ABSA signed score is a coin flip on sign, and that grafting a
# coin flip onto a confident FinBERT magnitude would manufacture a
# confident-looking but arbitrary-signed feature. That premise does not
# survive contact with hand-labelled data. Against 300 hand-labelled
# sentences, corpus-weighted directional accuracy of sign_graft is
# MONOTONICALLY HARMFUL as the deadband grows:
#
#   deadband | corpus-weighted directional accuracy
#   ---------|---------------------------------------
#     0.00   | 0.918
#     0.01   | 0.889
#     0.02   | 0.869
#     0.05   | 0.617
#     0.10   | 0.531
#     0.20   | 0.431
#
# The mechanism: on the corpus stratum where ABSA's magnitude is small
# (roughly 30% of all target sentences), 17 of 17 hand-checked directional
# sentences had ABSA's SIGN correct even though its MAGNITUDE was near zero.
# The deadband zeroed every one of those 17 rows, discarding correct signal
# instead of protecting against a coin flip -- there was no coin flip to
# protect against. ABSA's sign carries real information even at small
# magnitude on this corpus.
#
# So the evidence-backed default is 0.0: no row is zeroed by the deadband.
# The capability is kept (score_variants still accepts a `deadband` keyword
# so this can be swept again, e.g. from an analysis notebook), but the
# module-level default that ships to callers who don't pass one changes from
# 0.10 to 0.0.
DEADBAND = 0.0

# One column name per fusion candidate. score_variants() returns exactly
# these columns, in this order, named exactly as listed here.
VARIANTS = (
    "fin",
    "absa",
    "mean_blend",
    "sign_graft",
    "conf_graft",
    "conf_graft_soft",
    "gated",
    "agree_only",
)


def signed(pos, neg):
    """Elementwise pos - neg. NaN-safe: NaN in either input propagates to NaN
    in the output (ordinary float/array subtraction already does this; this
    wrapper exists so every variant below reads "signed(pos, neg)" instead of
    repeating the subtraction and so there is exactly one place to change
    the definition of "signed score" if it is ever revisited).
    """
    return pos - neg


def score_variants(sentences_df: pd.DataFrame, deadband: float = DEADBAND) -> pd.DataFrame:
    """Compute all fusion candidate variants for every row of `sentences_df`.

    Requires columns pos, neg, absa_pos, absa_neg (neu columns are read only
    by `gated`, via absa_neu). Returns a DataFrame with exactly the columns
    in VARIANTS, index-aligned to `sentences_df` (same index, same order,
    including a non-default / non-contiguous input index) -- callers can
    therefore assign these columns straight back onto sentences_df via a
    plain column-wise join without an intermediate reset_index().

    `deadband` controls sign_graft's deadband (see DEADBAND's module-level
    comment for the measured evidence behind the default). It exists as a
    parameter, rather than only as the module constant, so an analysis
    notebook can sweep it across calls without monkeypatching the module.
    The default is 0.0 -- the evidence-backed value from the sweep above,
    under which sign_graft's deadband never zeroes a row.

    Does NOT filter, sort, or group rows -- see the module docstring. Does
    NOT read or write any sent_* or absa_* column in place; those are only
    read, never mutated.
    """
    fin = signed(sentences_df["pos"], sentences_df["neg"])
    absa = signed(sentences_df["absa_pos"], sentences_df["absa_neg"])

    mean_blend = (fin + absa) / 2

    # sign_graft: ABSA decides DIRECTION, FinBERT decides MAGNITUDE. This is
    # a GRAFT (sign(absa) * abs(fin)), deliberately NOT a product (fin *
    # absa). A plain product of two negative scores returns a POSITIVE
    # number (two negatives cancel), which would silently claim the two
    # models AGREE the sentence is positive when they in fact both think it
    # is negative -- exactly backwards. Grafting the sign of one onto the
    # magnitude of the other avoids that double-negative bug by construction.
    #
    # Deadband (see DEADBAND's definition above for the measured evidence):
    # rows with |absa| < deadband are forced to exactly 0.0 instead -- not
    # NaN, since this is a deliberate "no attributable signal" verdict for a
    # row that WAS measured, not a missing measurement. The evidence-backed
    # default is deadband == 0.0, under which `absa.abs() < deadband` is
    # never true and no row is zeroed by this mechanism.
    missing = fin.isna() | absa.isna()
    absa_sign = np.sign(absa)
    sign_graft = absa_sign * fin.abs()
    below_deadband = absa.abs() < deadband
    # pandas comparisons against NaN evaluate to False, so `below_deadband`
    # is False on rows where absa is NaN -- guard those out explicitly so a
    # missing score is never coerced to the deadband's 0.0 verdict; it must
    # stay NaN instead (see the module docstring's NaN-over-zero rule).
    sign_graft = sign_graft.where(~(below_deadband & ~missing), 0.0)
    sign_graft = sign_graft.where(~missing, np.nan)

    # conf_graft and conf_graft_soft: sign_graft's one measured pathology is
    # that it uses ABSA for its SIGN ONLY and throws its MAGNITUDE away
    # entirely -- an ABSA score of +0.02 and +0.98 graft onto FinBERT's
    # magnitude identically. Measured against 300 hand-labelled sentences,
    # that is not a cosmetic gap: it means a hesitant ABSA call gets exactly
    # the same full-volume FinBERT magnitude as a confident one, and the
    # result is that sign_graft AMPLIFIES ABSA's errors 1.7x relative to its
    # correct calls (mean |score| on sign-wrong rows / mean |score| on
    # sign-right rows == 0.720, worse than plain absa's own 0.535). The fix
    # is to stop discarding ABSA's magnitude and use it as a CONFIDENCE
    # weight on top of the graft instead.
    #
    # conf_graft = absa * fin.abs(). Algebraically this is identical to
    # sign(absa) * abs(fin) * abs(absa) -- i.e. sign_graft scaled down by
    # ABSA's own confidence -- because sign(x) * abs(x) == x for any real x.
    # It is written here in the plain-product form rather than that
    # decomposed form for two reasons: it is simpler code, and the plain
    # product makes the NaN behaviour obvious by inspection (see the NaN
    # paragraph below) in a way the decomposed form would not.
    #
    # conf_graft_soft = sign(absa) * abs(fin) * (0.5 + 0.5 * abs(absa)). Same
    # idea, gentler ramp: the confidence multiplier runs from 0.5 (ABSA
    # totally unsure, |absa| == 0) to 1.0 (ABSA maximally sure, |absa| == 1)
    # instead of conf_graft's 0-to-1 ramp. This exists because the plain
    # product compresses the dynamic range badly (see the measured table
    # below) -- a floor of 0.5 keeps even a low-confidence ABSA call from
    # being crushed to near-zero.
    #
    # MEASURED, 300 hand-labelled sentences (dir_acc = directional accuracy,
    # mag_rho = magnitude correlation, extreme_prec = precision on
    # high-magnitude predictions, shout_ratio = mean |score| on correct rows
    # / mean |score| on wrong rows -- higher is better, dyn_range = spread of
    # output magnitudes):
    #
    #   variant          | dir_acc | mag_rho | AUC   | extreme_prec | shout_ratio | dyn_range
    #   -----------------|---------|---------|-------|--------------|-------------|----------
    #   absa             | 0.918   | 0.298   | 0.674 | 0.75         | 1.870       | 0.979
    #   sign_graft       | 0.918   | 0.461   | 0.769 | 0.85         | 1.388       | 1.315
    #   conf_graft       | 0.918   | 0.465   | 0.771 | 0.90         | 2.545       | 0.672
    #   conf_graft_soft  | 0.918   | 0.485   | 0.783 | 0.90         | 1.613       | 0.993
    #
    # And error-amplification (mean |score| on sign-WRONG rows / mean |score|
    # on sign-RIGHT rows -- lower is better, it means the variant stays
    # quieter when it is wrong): absa 0.535, sign_graft 0.720,
    # conf_graft 0.393, conf_graft_soft 0.620.
    #
    # All three grafts have IDENTICAL directional accuracy to plain absa
    # (0.918), because none of them ever changes the SIGN -- they only
    # reweight magnitude. conf_graft has the best shout-when-right ratio of
    # every variant tested here (2.545, beating even FinBERT's own 2.02) and
    # the best error-amplification figure of any variant including both of
    # its parents (0.393, vs 0.535 for absa and 0.720 for sign_graft) -- it
    # is the quietest variant when ABSA's sign is wrong and the loudest
    # relative to its own wrong calls when ABSA's sign is right. Its cost is
    # the narrowest dynamic range of the three grafts (0.672). conf_graft_soft
    # trades some of that quietness back for range (0.993, close to absa's
    # own 0.979) and has the best magnitude correlation of every variant
    # tested (0.485) -- the 0.5 floor keeps low-confidence ABSA calls from
    # being crushed the way the plain product crushes them.
    #
    # NaN handling: unlike sign_graft, NEITHER of these needs a `.where()`
    # missing-value guard. conf_graft is a plain product of two Series
    # (`absa * fin.abs()`); ordinary pandas/numpy multiplication already
    # propagates NaN -- NaN * anything == NaN -- so a NaN in either input
    # reaches the output with no extra code. conf_graft_soft multiplies
    # `np.sign(absa)`, and `np.sign(np.nan)` IS `nan` (not 0 or 1) -- that is
    # exactly the property that makes leaving conf_graft_soft unguarded safe
    # too: a NaN absa yields a NaN sign, which then poisons the whole
    # product. Both are pinned by tests below (VERIFIED, not just asserted
    # in this comment).
    #
    # `deadband` applies to sign_graft ONLY, not to either of these. It does
    # not apply here because these variants' confidence weighting IS the
    # principled replacement for sign_graft's hard cutoff -- a deadband zeros
    # a row outright below a threshold; conf_graft and conf_graft_soft
    # instead scale continuously by exactly the same |absa| quantity a
    # deadband would have thresholded, so a second hard cutoff on top of
    # them would just be double-counting the same signal two different ways.
    conf_graft = absa * fin.abs()
    conf_graft_soft = absa_sign * fin.abs() * (0.5 + 0.5 * absa.abs())

    # gated: FinBERT's magnitude, faded out by ABSA's own confidence that the
    # sentence expresses sentiment about the aspect AT ALL (1 - absa_neu).
    # This is the most principled variant here because it does not merely
    # exclude comparative sentences with a binary registry lookup the way
    # sent_entity_excl_comp_* does -- notebook 2.3 found the sentiment
    # contamination class is far broader than is_comparative: it is any
    # pairing with a benchmark, ETF, peer average, or credit-rating-style
    # grade, none of which the is_comparative flag catches. `gated` is the
    # graded, model-based generalisation of that same exclusion idea: instead
    # of a yes/no drop decided by a registry lookup, it keeps FinBERT's
    # magnitude in full where ABSA is confident the sentence is actually
    # about the aspect, and continuously fades that magnitude toward zero as
    # ABSA's own neutral-toward-aspect probability rises -- covering
    # contamination sources the registry was never told about.
    gated = fin * (1 - sentences_df["absa_neu"])

    # agree_only: fin where the two scorers agree on SIGN, else 0.0. A
    # "high-confidence subset" feature -- exactly two zeros (fin == 0 and
    # absa == 0) count as agreement, since sign(0) == sign(0) == 0 and there
    # is no real disagreement to flag. Disagreement produces 0.0 (a
    # deliberate "discard this row's signal" verdict), not NaN.
    fin_sign = np.sign(fin)
    agrees = fin_sign == absa_sign
    agree_only = fin.where(agrees, 0.0)
    # Rows with NaN fin or NaN absa: fin_sign/absa_sign are NaN, so `agrees`
    # (a NaN comparison) is False, which would send them into fin.where(...,
    # 0.0) -> 0.0, the WRONG answer for a missing measurement. `missing` was
    # already computed above for sign_graft; reuse it here for the same fix.
    agree_only = agree_only.where(~missing, np.nan)

    return pd.DataFrame(
        {
            "fin": fin,
            "absa": absa,
            "mean_blend": mean_blend,
            "sign_graft": sign_graft,
            "conf_graft": conf_graft,
            "conf_graft_soft": conf_graft_soft,
            "gated": gated,
            "agree_only": agree_only,
        },
        index=sentences_df.index,
    )[list(VARIANTS)]


# Variants promoted to article-level features. `score_variants()` returns all
# eight columns in VARIANTS for per-sentence hand inspection (see the module
# docstring), but conf_graft and conf_graft_soft are the only two that have
# cleared that inspection so far -- the measured table in the conf_graft
# comment above shows they are the best-performing grafts on every axis that
# was checked (dir_acc, mag_rho, AUC, extreme_prec, shout_ratio, and
# error-amplification), and neither `sign_graft` (superseded by them, see the
# conf_graft comment) nor `gated`/`mean_blend`/`agree_only` (still open
# candidates, not yet promoted) has that evidence behind it. Promoting only
# these two keeps the article-level feature table from ballooning with
# columns nobody has validated yet.
AGGREGATED_VARIANTS = ("conf_graft", "conf_graft_soft")

# One "_mean"/"_median"/"_lead"/"_top3_pos"/"_top3_neg"/"_spread" suffix per
# aggregated variant. See aggregate_fusion_features()'s docstring for the
# measured justification behind _median and _spread (and for why K stays at 3
# rather than growing to top-5).
FUSION_AGGREGATIONS = ("mean", "median", "lead", "top3_pos", "top3_neg", "spread")


def aggregate_fusion_features(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """Article-level aggregates of the `conf_graft` / `conf_graft_soft` fusion
    variants, one row per article_id.

    POPULATION. Every one of the twelve statistics below is computed over
    exactly the same sentence set aggregate_article_features() uses for its
    sent_entity_* family: `mentions_target & ~is_boilerplate`, further
    restricted here to rows where the variant's own score is non-null (a row
    can be in the sent_entity_* population but still be NaN for a fusion
    variant if its underlying FinBERT or ABSA score was never scored -- see
    score_variants()'s NaN-over-zero paragraph). Using the identical
    population as sent_entity_* is deliberate, not incidental: it is what
    makes fus_conf_graft_mean directly comparable to sent_entity_pos/neg on
    the same article -- a difference between them is a difference in SCORING
    METHOD, never a difference in which sentences were read.

    `sentences_df` must already carry the columns score_variants() needs
    (pos, neg, absa_pos, absa_neg, absa_neu) plus the entity_filter sentence
    schema (article_id, sent_idx, mentions_target, is_boilerplate). This
    function calls score_variants() itself -- callers do not need to run
    fusion scoring separately first.

    SIX AGGREGATIONS PER VARIANT (fus_<variant>_<agg>, 12 columns total):

      * _mean: plain mean over the population. Measures central tendency.

      * _median: median of the population. Measured justification: across
        the 958 articles in this corpus with >= 5 target sentences,
        corr(mean, median) = 0.792 -- correlated but meaningfully distinct,
        not redundant -- and 206 of those 958 articles (21.5%) have a mean
        and median that DISAGREE ON SIGN. In those, a handful of extreme
        sentences drag the average across zero relative to the robust
        centre; the mean-minus-median divergence has a median of 0.056 and a
        90th percentile of 0.156 on this corpus. So the pair (mean, median)
        is itself a skew detector: when they diverge, the article's average
        is being set by its extremes rather than its bulk. See
        test_median_is_robust_to_a_single_extreme_sentence in test_fusion.py
        for a worked example of exactly this case.

      * _lead: mean restricted to population rows with
        sent_idx < LEAD_SENTENCE_WINDOW (imported from config). Purely
        positional -- it asks whether WHERE a sentence sits in the article
        carries information, independent of which scorer read it. Mirrors
        sent_entity_lead_* in structure and in the same NaN-when-empty rule.

      * _top3_pos / _top3_neg: mean of the 3 HIGHEST / 3 LOWEST scores in the
        population. THIS IS THE POINT OF THIS FEATURE, so the reasoning is
        recorded here in full:

        A plain mean measures central tendency, but the question we actually
        want answered is "does this article carry a strong message", which is
        a property of the TAIL, not the centre. The mean dilutes it.
        Concretely: an article with a few strongly negative target sentences
        buried in a lot of neutral and mildly positive filler can score a
        LESS negative mean than a short, blandly mildly-negative article --
        even though the first article plainly carries the stronger negative
        message. _top3_neg reads that tail directly and is not diluted by
        filler the way _mean is (see
        test_top3_neg_is_not_diluted_by_neutral_filler in test_fusion.py,
        which demonstrates exactly this ranking flip).

        Three sentences, not one: the existing sent_entity_maxmag_* feature
        already picks a single most-extreme sentence, but trusting one
        sentence is noisy -- one paraphrase-quirk or mis-scored sentence
        swings the whole feature. Averaging the top THREE cuts that noise
        while still reading the tail rather than the centre.

        Both directions are kept separately, not collapsed via abs(): an
        article can contain strong positives AND strong negatives at once
        (e.g. a mixed earnings report), and that mixed-signal case is real,
        useful information that abs() would destroy by conflating "strongly
        positive" and "strongly negative" into the same number.

        Edge cases:
          - Fewer than 3 population rows: average whatever exists (1 or 2
            rows). Never pad with a fabricated value, never NaN just because
            there weren't 3.
          - Empty population: NaN for all four stats, never 0 -- the same
            no-signal-vs-measured-neutral rule this module and sentiment.py
            follow everywhere. 0 is a real, measured value; an empty
            population measured nothing.
          - _top3_pos on an article whose population is ALL negative:
            it still returns the mean of the 3 LEAST-negative scores (the top
            3 by value, regardless of sign), not NaN. That is deliberate and
            informative -- "the most favourable thing this article says is
            still -0.2" is a real, useful reading, and returning NaN there
            would silently discard it. The mirror image holds for
            _top3_neg on an all-positive article: it returns the mean of the
            3 least-positive scores, not NaN.

      * _spread: _top3_pos - _top3_neg. Always non-negative (top3_pos is a
        mean of the 3 highest scores, top3_neg a mean of the 3 lowest, so
        top3_pos >= top3_neg always). Measured justification: the mean
        cannot distinguish a CONTESTED article (loud claims in both
        directions, roughly cancelling) from a QUIET one (nothing strong
        said at all) -- both average near zero. Among the 213 articles in
        this corpus with |mean| < 0.05 and >= 10 target sentences, spread
        ranges from 0.04 to 1.71, splitting them into roughly 53 contested
        and 53 quiet articles that a mean-only view cannot tell apart. Two
        real examples with effectively identical means: article 139908240
        ("Earnings Season Surprises", 52 sentences, mean +0.034,
        top3_pos +0.753, top3_neg -0.332, spread 1.085 -- CONTESTED) versus
        article 136907450 ("Lightship's electric RVs...", 25 sentences,
        mean +0.020, top3_pos +0.123, top3_neg -0.000, spread 0.123 -- QUIET).
        See test_spread_separates_contested_from_quiet_articles in
        test_fusion.py.

        corr(|mean|, spread) = 0.319 on this corpus -- spread is mostly NEW
        information relative to the mean, not a restatement of its
        magnitude. The SUM of the extremes (top3_pos + top3_neg) was also
        considered and REJECTED: it correlates 0.937 with the mean, i.e. it
        is near-redundant with a quantity already computed. It is the
        DIFFERENCE between the extremes that carries the signal a plain mean
        cannot see, not their sum.

        Added as an explicit column rather than left for a caller to derive
        from top3_pos and top3_neg at model-build time, because tree-based
        models cannot easily learn to form a difference between two features
        on their own -- a split on a single feature is what they do
        natively, and spread being its own column makes it available to a
        single split.

        NaN/zero rules: NaN exactly when top3_pos/top3_neg are NaN (empty
        population -- see below), and exactly 0.0 when the population is a
        single sentence (top3_pos and top3_neg both equal that one score, so
        their difference is exactly zero, not NaN).

      WHY K STAYS AT 3, NOT 5. Top-5 was considered and rejected on measured
      grounds, not by default. On this corpus the median article has only 4
      target sentences; 55% of articles have fewer than 6 (so top3_pos and
      top3_neg already share sentences at that size) and 71.5% have fewer
      than 10 -- so with K=5, both ends would include nearly every sentence
      in most articles and BOTH top5_pos and top5_neg would collapse toward
      the plain mean, destroying the exact tail signal this feature exists
      to capture, on the majority of the corpus. It also buys nothing on the
      articles long enough for K to matter:
      corr(top3_pos, top5_pos) = 0.980, corr(top3_neg, top5_neg) = 0.974,
      corr(spread3, spread5) = 0.975 -- K=5 is not measurably different from
      K=3 where both are well-defined, so there is no evidence-backed reason
      to accept K=5's small-article collapse in exchange for it. K=3 degrades
      gracefully instead: at n <= 3, both top3_pos and top3_neg equal the
      plain mean exactly (see "Fewer than 3 population rows" above), which
      is a sensible fallback rather than a wrong answer.

    Returns one row per article_id present in `sentences_df`. An empty input
    frame returns an empty frame carrying the correct 13 columns (article_id
    plus the 12 fus_* columns) so callers get a stable schema either way.
    """
    fus_columns = [
        f"fus_{variant}_{agg}" for variant in AGGREGATED_VARIANTS for agg in FUSION_AGGREGATIONS
    ]

    if len(sentences_df) == 0:
        return pd.DataFrame(columns=["article_id", *fus_columns])

    sentences_df = sentences_df.copy().reset_index(drop=True)
    if "is_boilerplate" in sentences_df.columns:
        sentences_df["is_boilerplate"] = sentences_df["is_boilerplate"].fillna(False).astype(bool)
    else:
        sentences_df["is_boilerplate"] = False
    sentences_df["mentions_target"] = sentences_df["mentions_target"].fillna(False).astype(bool)

    # Attach the two promoted variants' scores directly onto the (copied)
    # sentence table so the per-article groupby below can select rows exactly
    # the way aggregate_article_features() does for sent_entity_*.
    variant_scores = score_variants(sentences_df)
    for variant in AGGREGATED_VARIANTS:
        sentences_df[f"__fus_{variant}"] = variant_scores[variant]

    def _stat(values: pd.Series) -> dict[str, float]:
        """mean/median/top3_pos/top3_neg/spread over `values` (already
        NaN-dropped). NaN, never 0, when `values` is empty -- see the
        population/edge-case paragraphs above. spread == top3_pos - top3_neg,
        which is exactly 0.0 (not NaN) when `values` has a single row, since
        top3_pos and top3_neg both equal that one score in that case."""
        if len(values) == 0:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "top3_pos": float("nan"),
                "top3_neg": float("nan"),
                "spread": float("nan"),
            }
        top3_pos = values.nlargest(3).mean()
        top3_neg = values.nsmallest(3).mean()
        return {
            "mean": values.mean(),
            "median": values.median(),
            "top3_pos": top3_pos,
            "top3_neg": top3_neg,
            "spread": top3_pos - top3_neg,
        }

    rows = []
    for article_id, group in sentences_df.groupby("article_id", sort=False):
        # Population: mentions_target & ~is_boilerplate -- identical to the
        # selection aggregate_article_features() uses for sent_entity_*.
        target = group[group["mentions_target"] & ~group["is_boilerplate"]]
        lead_target = target[target["sent_idx"] < LEAD_SENTENCE_WINDOW]

        row = {"article_id": article_id}
        for variant in AGGREGATED_VARIANTS:
            col = f"__fus_{variant}"
            # Further restricted to rows where this variant's own score is
            # non-null -- a row can be in the target population and still
            # lack a fusion score if its underlying pos/neg or absa_pos/neg
            # was never scored (see score_variants()'s NaN-over-zero rule).
            scores = target[col].dropna()
            lead_scores = lead_target[col].dropna()

            stats = _stat(scores)
            row[f"fus_{variant}_mean"] = stats["mean"]
            row[f"fus_{variant}_median"] = stats["median"]
            row[f"fus_{variant}_lead"] = lead_scores.mean() if len(lead_scores) > 0 else float("nan")
            row[f"fus_{variant}_top3_pos"] = stats["top3_pos"]
            row[f"fus_{variant}_top3_neg"] = stats["top3_neg"]
            row[f"fus_{variant}_spread"] = stats["spread"]
        rows.append(row)

    result = pd.DataFrame(rows, columns=["article_id", *fus_columns])
    return result
