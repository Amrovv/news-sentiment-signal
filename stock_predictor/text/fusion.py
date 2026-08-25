"""Per-sentence fusions of FinBERT and ABSA sentiment.

FinBERT scores the sentence and is finance-trained; ABSA scores toward an aspect
and is the only one told which company the sentence is supposed to be about. This
module combines them into candidate per-sentence scores and aggregates the
promoted ones to article level.

Reads pos/neg/neu and absa_pos/absa_neg/absa_neu, writes only new columns.

NaN over zero throughout: 0.0 is a real value here, produced by sign_graft's
deadband and agree_only's disagreement case, so a row missing either model's
score is NaN in every variant rather than 0.

Selection of variants, floors and aggregations is argued in notebooks/text/2.2
and 2.3; this module only implements the result.
"""

import numpy as np
import pandas as pd

from stock_predictor.config import LEAD_SENTENCE_WINDOW

# |absa| below this is treated as too near neutral to trust for direction, and
# sign_graft zeroes the row. Swept in notebooks/; 0.0 measured best.
DEADBAND = 0.0

# Floor of the conf_graft family:
#     sign(absa) * abs(fin) * (floor + (1 - floor) * abs(absa))
# 0.0 is conf_graft, 0.5 conf_graft_soft, 1.0 sign_graft. This is the shipped
# value; see notebooks/fusion-weight-sweep.txt for the sweep behind it.
CONF_FLOOR = 0.7

# Small-sample shrinkage: shrunk = raw * n / (n + K). Pulls population means
# built from few sentences toward zero, fixing single-sentence articles
# spuriously landing at the extremes. Fixed constants, not refit per corpus,
# so build_model_features() (batch) and analyze() (live) agree. Only the
# two population-mean columns below benefit; conf_graft_floor_lead,
# fus_lead_gap, and fus_ceo_mean got worse under shrinkage and are left
# alone. See notebooks/2.1-aw-extended-merged-eda.ipynb sections 9.1, 9.2,
# 10, 10.1 for the sweep and per-column results.
SHRINKAGE_K_ENTITY = 4.0
SHRINKAGE_K_TRUSTED = 4.0


def shrink(raw, n, k):
    """Small-sample shrinkage: raw * n / (n + k). n=0 -> 0.0, not NaN."""
    n = pd.Series(n).astype(float)
    factor = n / (n + k)
    return raw * factor


# One column name per fusion candidate. score_variants() returns exactly
# these columns, in this order, named exactly as listed here.
VARIANTS = (
    "fin",
    "absa",
    "mean_blend",
    "sign_graft",
    "conf_graft",
    "conf_graft_soft",
    "conf_graft_floor",
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
    """Compute every fusion candidate for each row of `sentences_df`.

    Requires pos, neg, absa_pos, absa_neg; absa_neu is read only by `gated`.

    Returns the VARIANTS columns, index-aligned to the input, so callers can assign
    them straight back. Does not filter, sort or group rows, and mutates nothing.
    """
    fin = signed(sentences_df["pos"], sentences_df["neg"])
    absa = signed(sentences_df["absa_pos"], sentences_df["absa_neg"])

    mean_blend = (fin + absa) / 2

    missing = fin.isna() | absa.isna()
    absa_sign = np.sign(absa)
    sign_graft = absa_sign * fin.abs()
    below_deadband = absa.abs() < deadband
    # NaN compares False, so an unscored row would be coerced to the 0.0 verdict.
    sign_graft = sign_graft.where(~(below_deadband & ~missing), 0.0)
    sign_graft = sign_graft.where(~missing, np.nan)

    conf_graft = absa * fin.abs()
    conf_graft_soft = absa_sign * fin.abs() * (0.5 + 0.5 * absa.abs())

    conf_graft_floor = absa_sign * fin.abs() * (CONF_FLOOR + (1.0 - CONF_FLOOR) * absa.abs())

    gated = fin * (1 - sentences_df["absa_neu"])

    # Two zeros count as agreement. Disagreement gives 0.0, a discard verdict.
    fin_sign = np.sign(fin)
    agrees = fin_sign == absa_sign
    agree_only = fin.where(agrees, 0.0)
    # NaN compares False, so an unscored row would fall through to 0.0.
    agree_only = agree_only.where(~missing, np.nan)

    return pd.DataFrame(
        {
            "fin": fin,
            "absa": absa,
            "mean_blend": mean_blend,
            "sign_graft": sign_graft,
            "conf_graft": conf_graft,
            "conf_graft_soft": conf_graft_soft,
            "conf_graft_floor": conf_graft_floor,
            "gated": gated,
            "agree_only": agree_only,
        },
        index=sentences_df.index,
    )[list(VARIANTS)]


# Variants aggregated to article level. score_variants() returns all of VARIANTS
# per sentence; only these reach the feature table.
AGGREGATED_VARIANTS = ("conf_graft_floor",)
FUSION_AGGREGATIONS = ("mean", "median", "lead", "top3_pos", "top3_neg", "spread")


def aggregate_fusion_features(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """Article-level aggregates of each promoted variant, one row per article_id.

    Population is `mentions_target & ~is_boilerplate`, restricted per variant to rows
    whose own score is non-null. Identical to aggregate_article_features()'s
    sent_entity_* population, so the two are directly comparable.

    Calls score_variants() itself; `sentences_df` needs only its inputs plus the
    entity_filter schema.

    Six aggregations per variant, fus_<variant>_<agg>: mean, median, lead (over
    sent_idx < LEAD_SENTENCE_WINDOW), top3_pos, top3_neg, and spread
    (top3_pos - top3_neg).

    Fewer than three population rows averages what exists. An empty population is NaN
    throughout, never 0; _lead is the exception and is 0.0. _spread is 0.0 for a
    single-sentence population.

    Returns a stable schema on empty input.
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
            # 0.0, not NaN: an article with no early target mention delivers no
            # early sentiment, which is a measurement rather than an absence.
            row[f"fus_{variant}_lead"] = lead_scores.mean() if len(lead_scores) > 0 else 0.0
            row[f"fus_{variant}_top3_pos"] = stats["top3_pos"]
            row[f"fus_{variant}_top3_neg"] = stats["top3_neg"]
            row[f"fus_{variant}_spread"] = stats["spread"]
        rows.append(row)

    result = pd.DataFrame(rows, columns=["article_id", *fus_columns])
    return result


# The three ways a sentence can end up in the target population, in the
# precedence order entity_filter applies them. These PARTITION the population --
# every target sentence lands in exactly one channel -- which is what makes the
# counts below safe to sum and safe to reason about as shares.
#
# A sentence with resolved_by_coref unset named the company explicitly; one with
# it set was resolved by the model, and the presence of a mention span splits
# that into the two coref channels.
PROVENANCE_CHANNELS = ("surface", "coref_span", "coref_nospan")

# Referent accuracy differs sharply by channel, which is why they are kept separate
# rather than pooled. See notebooks/text/2.1 sections 4 and 5.


def provenance_channel(sentences_df: pd.DataFrame) -> pd.Series:
    """Label each row with the channel that put it in the target population.

    Returns a Series aligned to `sentences_df`, None outside the population. Missing
    provenance columns read as all-False, so a table with no resolution evidence reads
    as all `surface`.

    Order is precedence: `surface` first, then overwritten for rows coref spoke for.
    """
    n = len(sentences_df)
    mentions = sentences_df.get("mentions_target", pd.Series(False, index=sentences_df.index))
    mentions = mentions.fillna(False).astype(bool)

    def _flag(name):
        col = sentences_df.get(name, pd.Series(False, index=sentences_df.index))
        return col.fillna(False).astype(bool)

    by_coref = _flag("resolved_by_coref")
    if "mention_char_start" in sentences_df.columns:
        has_span = sentences_df["mention_char_start"].notna()
    else:
        has_span = pd.Series(False, index=sentences_df.index)

    channel = pd.Series([None] * n, index=sentences_df.index, dtype=object)
    channel[mentions] = "surface"
    channel[mentions & by_coref & has_span] = "coref_span"
    channel[mentions & by_coref & ~has_span] = "coref_nospan"
    return channel


def aggregate_provenance_features(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """Article-level counts and means split by how each sentence was tagged.

    Three columns per channel in PROVENANCE_CHANNELS:

        prov_<channel>_n               population sentences in the channel
        prov_<channel>_share           that count over the article's target population
        prov_<channel>_<variant>_mean  the channel's own mean, per AGGREGATED_VARIANTS

    Population is identical to aggregate_fusion_features(), so the channel means are
    comparable to fus_<variant>_mean. Purely additive: no fus_* column is touched.

    Empty channels are NaN, never 0. Counts are the exception and are 0.

    Returns a stable schema on empty input.
    """
    prov_columns = []
    for channel in PROVENANCE_CHANNELS:
        prov_columns.append(f"prov_{channel}_n")
        prov_columns.append(f"prov_{channel}_share")
        for variant in AGGREGATED_VARIANTS:
            prov_columns.append(f"prov_{channel}_{variant}_mean")

    if len(sentences_df) == 0:
        return pd.DataFrame(columns=["article_id", *prov_columns])

    sentences_df = sentences_df.copy().reset_index(drop=True)
    if "is_boilerplate" in sentences_df.columns:
        sentences_df["is_boilerplate"] = sentences_df["is_boilerplate"].fillna(False).astype(bool)
    else:
        sentences_df["is_boilerplate"] = False
    sentences_df["mentions_target"] = sentences_df["mentions_target"].fillna(False).astype(bool)

    sentences_df["__channel"] = provenance_channel(sentences_df)
    variant_scores = score_variants(sentences_df)
    for variant in AGGREGATED_VARIANTS:
        sentences_df[f"__fus_{variant}"] = variant_scores[variant]

    rows = []
    for article_id, group in sentences_df.groupby("article_id", sort=False):
        target = group[group["mentions_target"] & ~group["is_boilerplate"]]
        n_target = len(target)

        row = {"article_id": article_id}
        for channel in PROVENANCE_CHANNELS:
            in_channel = target[target["__channel"] == channel]
            row[f"prov_{channel}_n"] = len(in_channel)
            # NaN share for an article with no target sentences at all: there
            # is no denominator, and 0.0 would claim the channel was measured
            # and found empty rather than that nothing was measured.
            row[f"prov_{channel}_share"] = len(in_channel) / n_target if n_target else float("nan")
            for variant in AGGREGATED_VARIANTS:
                scores = in_channel[f"__fus_{variant}"].dropna()
                row[f"prov_{channel}_{variant}_mean"] = (
                    scores.mean() if len(scores) > 0 else float("nan")
                )
        rows.append(row)

    return pd.DataFrame(rows, columns=["article_id", *prov_columns])


# Article-level fusion features that are NOT aggregations over the target
# sentence population: one over the CEO-mention population, one over the
# headline. Both are the same graft at CONF_FLOOR, so they are directly
# comparable with fus_conf_graft_floor_mean.
CEO_FUSION_COLUMNS = ["fus_ceo_mean"]
HEADLINE_FUSION_COLUMNS = ["fus_headline"]


def fuse(pos, neg, absa_pos, absa_neg, floor: float = CONF_FLOOR):
    """The shipped graft, applied to raw probability columns.

    sign(absa) * abs(fin) * (floor + (1 - floor) * abs(absa)), with fin = pos - neg
    and absa = absa_pos - absa_neg. Elementwise on Series or scalars.

    For callers needing one fused number without the full variant sweep: the CEO
    population and the headline. NaN propagates.
    """
    fin = signed(pos, neg)
    absa = signed(absa_pos, absa_neg)
    return np.sign(absa) * fin.abs() * (floor + (1.0 - floor) * absa.abs())


def aggregate_ceo_fusion_features(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """fus_ceo_mean, one row per article_id.

    Population is `mentions_ceo & ~is_boilerplate` with a non-null fused score,
    matching sent_ceo_*. One statistic rather than six, since the population is empty
    for ~70% of articles. NaN when empty.
    """
    df = sentences_df.copy()
    if "is_boilerplate" in df.columns:
        df["is_boilerplate"] = df["is_boilerplate"].fillna(False).astype(bool)
    else:
        df["is_boilerplate"] = False

    df["_fused"] = fuse(df["pos"], df["neg"], df["absa_pos"], df["absa_neg"])
    pop = df[df["mentions_ceo"].fillna(False) & ~df["is_boilerplate"] & df["_fused"].notna()]

    ids = pd.Index(sentences_df["article_id"].unique(), name="article_id")
    means = pop.groupby("article_id")["_fused"].mean()
    return pd.DataFrame({"fus_ceo_mean": means.reindex(ids)}).reset_index()


def headline_fusion_feature(
    headline_finbert: pd.DataFrame, headline_absa: pd.DataFrame
) -> pd.DataFrame:
    """fus_headline, one row per article_id.

    Takes sentiment.score_headlines() output (article_id, pos, neg, neu) and
    absa.score_headlines() output (article_id, absa_pos, absa_neg, absa_neu)
    and grafts them exactly as the sentence scorer does.

    A headline is one string with no preceding context, so there is no
    aggregation to do -- one headline, one number. NaN if either scorer is
    missing for that article.
    """
    merged = headline_finbert[["article_id", "pos", "neg"]].merge(
        headline_absa[["article_id", "absa_pos", "absa_neg"]], on="article_id", how="outer"
    )
    merged["fus_headline"] = fuse(
        merged["pos"], merged["neg"], merged["absa_pos"], merged["absa_neg"]
    )
    return merged[["article_id", "fus_headline"]]


EXTRA_FUSION_COLUMNS = ["fus_maxmag", "fus_trusted_mean", "fus_scorer_gap"]

# Channels trusted for fus_trusted_mean. coref_nospan is excluded as the weakest
# measured channel; see notebooks/text/2.1 section 5.
TRUSTED_CHANNELS = ("surface", "coref_span")


def aggregate_extra_fusion_features(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """Three further article-level features derived from the shipped fusion score.

    Population is aggregate_fusion_features()'s, so these are comparable to the
    fus_<variant>_* columns on the same article.

        fus_maxmag        the signed score of the single loudest target sentence,
                          chosen by largest absolute fused score. The tail that
                          _top3_pos and _top3_neg average over, undiluted.
        fus_trusted_mean  the mean over TRUSTED_CHANNELS only. Folds the
                          provenance split into one number rather than nine.
        fus_scorer_gap    mean |fin - absa| over the population: how far apart the
                          two scorers were, which no fused score records.

    Also returns n_trusted_sents (the trusted-channel population size behind
    fus_trusted_mean) as an internal column, not part of EXTRA_FUSION_COLUMNS
    since it's not a model feature itself — build_model_features() consumes
    it for shrinkage and drops it.

    NaN, never 0, on an empty population. n_trusted_sents is always a count
    (0, not NaN, on an empty population).
    """
    if len(sentences_df) == 0:
        return pd.DataFrame(columns=["article_id", *EXTRA_FUSION_COLUMNS, "n_trusted_sents"])

    df = sentences_df.copy().reset_index(drop=True)
    if "is_boilerplate" in df.columns:
        df["is_boilerplate"] = df["is_boilerplate"].fillna(False).astype(bool)
    else:
        df["is_boilerplate"] = False
    df["mentions_target"] = df["mentions_target"].fillna(False).astype(bool)

    variants = score_variants(df)
    shipped = AGGREGATED_VARIANTS[0]
    df["__fus"] = variants[shipped]
    df["__gap"] = (variants["fin"] - variants["absa"]).abs()
    df["__channel"] = provenance_channel(df)

    pop = df[df["mentions_target"] & ~df["is_boilerplate"]]

    rows = []
    for article_id, g in pop.groupby("article_id", sort=True):
        scored = g[g["__fus"].notna()]
        trusted = scored[scored["__channel"].isin(TRUSTED_CHANNELS)]
        gaps = g["__gap"].dropna()
        rows.append(
            {
                "article_id": article_id,
                "fus_maxmag": (
                    scored.loc[scored["__fus"].abs().idxmax(), "__fus"]
                    if len(scored)
                    else float("nan")
                ),
                "fus_trusted_mean": trusted["__fus"].mean() if len(trusted) else float("nan"),
                "fus_scorer_gap": gaps.mean() if len(gaps) else float("nan"),
                "n_trusted_sents": len(trusted),
            }
        )

    out = pd.DataFrame(rows, columns=["article_id", *EXTRA_FUSION_COLUMNS, "n_trusted_sents"])
    return out
