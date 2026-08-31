from dataclasses import dataclass
import re

import pandas as pd

from stock_predictor.config import INTERIM_DATA_DIR

# Verbs which indicate price movement discussion
MOVE_VERBS = (
    r"(rises?|falls?|slides?|slips?|surges?|jumps?|climbs?|tumbles?|sinks?|dips?|sank|sunk|"
    r"gains?|drops?|plunges?|soars?|rallies?|retreats?|declines?|advances?|"
    r"rebounds?|bounces?|rockets?|spikes?|"
    r"rose|fell|slid|slipped|surged|jumped|climbed|tumbled|sank|"
    r"gained|dropped|plunged|soared|rallied|retreated|declined|advanced|"
    r"rebounded|bounced|rocketed|spiked)"
)

# Strong: near-certain price-commentary signals
STRONG_PATTERNS = [
    # (?<!market\s) excludes "market share(s)", a business metric, not a price move.
    rf"(?<!market\s)\bshares?\s+(?:\w+\s+){{0,2}}{MOVE_VERBS}",
    rf"\bstocks?\s+(?:\w+\s+){{0,2}}{MOVE_VERBS}",
    r"\bstocks?\s+(closed|opened|traded|finished)\b",
    r"\b(up|down|off)\s+\d+(\.\d+)?\s*%",
    r"\b\d+(\.\d+)?\s*%\s+(higher|lower|gain|loss|drop|jump|surge|rally|slide|tumble|plunge|rebound)\b",
    r"\bextend(s|ed|ing)?\s+(gains|losses|slide|rally)\b",
    r"\bhit(s|ting)?\s+(a\s+)?(new\s+)?(record|all-time)\s+(high|low)\b",
    r"\b(52|fifty-two)[- ]week\s+(high|low)\b",
]

# Weak: suggestive but common in analysis pieces too
WEAK_PATTERNS = [
    r"\b(rally|selloff|sell-off|slump|surge|plunge|rout|rebound)\b",
    r"\bhere'?s\s+why\b",
    r"\binvestors?\s+(weighed|reacted|shrugged|cheered|punished)\b",
    # Dropped the "the market/s&p/nasdaq" requirement: "outperformed" alone is
    # already common-in-analysis-pieces territory, the weak tier's own bar.
    r"\b(outperform|underperform)(ed|ing)?\b",
    r"\b(premarket|pre-market|after[- ]hours)\s+(trading|movers?)\b",
    r"\bmoving\s+(the\s+)?(market|stock)s?\b",
    r"\b(top|biggest)\s+(gainers?|losers?|movers?)\b",
    r"\bwhat'?s\s+(driving|behind)\b",
    # A direction with no magnitude ("stock is up") is weaker evidence than a
    # percentage, so it stays out of STRONG_PATTERNS.
    r"\b(shares?|stocks?)\s+(?:is|are|was|were)\s+(up|down)\b",
]

_STRONG = [re.compile(p, re.IGNORECASE) for p in STRONG_PATTERNS]
_WEAK = [re.compile(p, re.IGNORECASE) for p in WEAK_PATTERNS]


@dataclass
class ReactiveResult:
    is_reactive: int  # 0/1 reactive flag
    score: float  # Score based on the hits
    hits: list[str]  # Which patterns fired (for debugging/tuning)


def check_hits(text: str, patterns) -> list[str]:
    """
    Returns a list of hits from the given text

    text: string which contains summary/headline to be checked
    patterns: list of patterns that we are looking for
    """

    if not text:
        return []
    return [p.search(text).group() for p in patterns if p.search(text)]


def classify_reactive(
    headline: str, summary: str = "", article: str = "", threshold: float = 1.0
) -> ReactiveResult:
    """
    Classify a whole article.

    Weights: headline strong=1.0, headline weak=0.5,
             summary strong=0.5, summary weak=0.25,
             article strong=0.2, article weak=0.1,
    Default threshold: 1.0 -> a single strong headline hit is enough.
    """
    # Pull hits
    h_strong = check_hits(headline, _STRONG)
    h_weak = check_hits(headline, _WEAK)
    s_strong = check_hits(summary, _STRONG)
    s_weak = check_hits(summary, _WEAK)
    a_strong = check_hits(article, _STRONG)
    a_weak = check_hits(article, _WEAK)

    # Score the hits accordingly
    score = (
        1.00 * len(h_strong)
        + 0.50 * len(h_weak)
        + 0.50 * len(s_strong)
        + 0.25 * len(s_weak)
        + 0.20 * len(a_strong)
        + 0.10 * len(a_weak)
    )

    # H = headline, S = summary, A = article
    # ! = strong, ? = weak
    matched = (
        [f"H! {p}" for p in h_strong]
        + [f"H? {p}" for p in h_weak]
        + [f"S! {p}" for p in s_strong]
        + [f"S? {p}" for p in s_weak]
        + [f"A! {p}" for p in a_strong]
        + [f"A? {p}" for p in a_weak]
    )
    return ReactiveResult(int(score >= threshold), score, matched)


def classify_reactive_sentence(text: str, threshold: float = 1.0) -> ReactiveResult:
    """
    Classify a single sentence.

    A sentence carries no headline/summary/article tiering, so both pattern
    sets are weighted at the headline tier (strong=1.0, weak=0.5) -- a single
    strong hit is a direct price-move claim regardless of where in the
    article the sentence sits.
    """
    strong = check_hits(text, _STRONG)
    weak = check_hits(text, _WEAK)
    score = 1.00 * len(strong) + 0.50 * len(weak)
    matched = [f"! {p}" for p in strong] + [f"? {p}" for p in weak]
    return ReactiveResult(int(score >= threshold), score, matched)


def tag_reactive_sentences(sentences_df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Tag each row of a sentence table (entity_filter.SENTENCE_COLUMNS schema)
    with reactivity score/flag. Tagging only -- mirrors entity_filter's own
    "does not filter sentences out" convention, so downstream aggregation
    decides what to do with boilerplate/irrelevant rows.
    """
    out = sentences_df.copy()
    if len(out):
        results = out["text"].apply(lambda t: classify_reactive_sentence(t, threshold=threshold))
        out["is_reactive"] = [r.is_reactive for r in results]
        out["reactive_score"] = [r.score for r in results]
        out["reactive_hits"] = [r.hits for r in results]
    else:
        out["is_reactive"] = pd.Series(dtype=int)
        out["reactive_score"] = pd.Series(dtype=float)
        out["reactive_hits"] = pd.Series(dtype=object)
    return out


def aggregate_reactive_features(
    sentences_df: pd.DataFrame, headlines_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Article-level reactivity aggregates from a sentence table already tagged by
    tag_reactive_sentences() (must carry is_reactive/reactive_score).

    Aggregated over the same target population sentiment.aggregate_article_features()
    scores over -- mentions_target, non-boilerplate -- not every sentence in the
    article. A reactive sentence about a different company mentioned in passing
    ("Meanwhile, Ford's shares surged 8%...") should not count toward this article's
    own reactivity, and restricting to target sentences is what keeps it out; a
    genuinely target-referring sentence resolved only through coreference
    ("The company's stock fell 5%...") is still included, since entity_filter's coref
    resolution already sets mentions_target=True for it.

    Headline reactivity is kept as its own column rather than blended into the body
    mean at a hand-picked weight. This project's fusion layer already treats headline
    and body evidence as separate features a downstream model weighs itself
    (fus_headline vs. fus_conf_graft_floor_mean, kept apart precisely so a tree model
    can learn its own split rather than have one weighting baked in ahead of time) --
    there is no reason for reactivity to be the one feature family that pre-decides
    that trade-off instead of learning it the same way.

    n_reactive_sents is always a real count, 0 on an empty target population, the same
    convention n_entity_sents uses. reactive_share/mean/max are NaN on an empty target
    population, the same convention sent_entity_pos/neg/neu use -- "no target sentences"
    and "target sentences, none reactive" are different claims, and only the second one
    is a real 0.
    """
    required = {"is_reactive", "reactive_score"}
    missing = required - set(sentences_df.columns)
    if missing:
        raise ValueError(
            f"aggregate_reactive_features: sentences_df is missing {sorted(missing)}; "
            "run tag_reactive_sentences() first."
        )
    if "mentions_target" not in sentences_df.columns:
        raise ValueError("aggregate_reactive_features: sentences_df has no mentions_target column")

    if len(sentences_df) == 0:
        out = pd.DataFrame(
            columns=[
                "article_id",
                "n_reactive_sents",
                "reactive_share",
                "reactive_mean",
                "reactive_max",
            ]
        )
    else:
        df = sentences_df.copy().reset_index(drop=True)
        df["mentions_target"] = df["mentions_target"].fillna(False).astype(bool)
        if "is_boilerplate" in df.columns:
            df["is_boilerplate"] = df["is_boilerplate"].fillna(False).astype(bool)
        else:
            df["is_boilerplate"] = False

        rows = []
        for article_id, group in df.groupby("article_id", sort=False):
            target = group[group["mentions_target"] & ~group["is_boilerplate"]]
            n_target = len(target)
            if n_target > 0:
                n_reactive = int(target["is_reactive"].sum())
                rows.append(
                    {
                        "article_id": article_id,
                        "n_reactive_sents": n_reactive,
                        "reactive_share": n_reactive / n_target,
                        "reactive_mean": target["reactive_score"].mean(),
                        "reactive_max": target["reactive_score"].max(),
                    }
                )
            else:
                rows.append(
                    {
                        "article_id": article_id,
                        "n_reactive_sents": 0,
                        "reactive_share": float("nan"),
                        "reactive_mean": float("nan"),
                        "reactive_max": float("nan"),
                    }
                )
        out = pd.DataFrame(rows)

    if headlines_df is not None:
        headline_reactive = headlines_df[["article_id", "headline"]].copy()
        headline_reactive["reactive_headline"] = headline_reactive["headline"].apply(
            lambda h: classify_reactive_sentence(h if isinstance(h, str) else "").score
        )
        out = out.merge(
            headline_reactive[["article_id", "reactive_headline"]], on="article_id", how="left"
        )
    else:
        out["reactive_headline"] = float("nan")

    return out


def threshold_analysis(
    labelled_path: str,
    sample_df: pd.DataFrame,
    thresholds: list[float] | None = None,
    article_column: str = "article",
) -> pd.DataFrame:
    """
    Run evaluate_sample() across a threshold grid and return the resulting metrics.
    """
    if thresholds is None:
        thresholds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    rows = []
    for thresh in thresholds:
        metrics = evaluate_sample(
            labelled_path,
            sample_df,
            threshold=thresh,
            article_column=article_column,
        )
        rows.append(
            {
                "threshold": thresh,
                "agreement": metrics["agreement"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "base_rate_pred": metrics["base_rate_pred"],
                "base_rate_true": metrics["base_rate_true"],
                "tp": metrics["confusion"]["tp"],
                "fp": metrics["confusion"]["fp"],
                "fn": metrics["confusion"]["fn"],
                "tn": metrics["confusion"]["tn"],
            }
        )

    return pd.DataFrame(rows)


def make_labelling_sample(article_df: pd.DataFrame, seed: int, n: int = 100):
    """
    Draw a random sample and write a CSV with a blank `true_label` column.
    Fill it in by hand (1 = reactive, 0 = not), then feed it to evaluate_sample().

    Deliberately omits the predicted flag so the hand labels stay unbiased.
    """
    path = INTERIM_DATA_DIR / "reactive_sample.csv"
    cols = ["article_id", "headline", "summary"]
    if "processed_body" in article_df.columns:
        cols.append("processed_body")
    elif "article" in article_df.columns:
        cols.append("article")

    sample = article_df.sample(n=min(n, len(article_df)), random_state=seed)[cols].copy()
    sample["hand_label"] = ""
    sample.to_csv(path, index=False)
    print(f"Wrote {len(sample)} rows to {path} — fill in true_label, then run evaluate().")
    return sample


def evaluate_sample(
    labelled_path: str,
    sample_df: pd.DataFrame,
    threshold: float = 1.0,
    article_column: str = "article",
) -> dict:
    """
    Evaluate the hand labelled articles against our algorithm.
    """
    labelled_sample = pd.read_csv(labelled_path)
    labelled_sample = labelled_sample[
        labelled_sample["hand_label"].notna() & (labelled_sample["hand_label"] != "")
    ]
    labelled_sample["hand_label"] = labelled_sample["hand_label"].astype(int)

    def _article_text(r):
        """Check the right column name for our full article text"""
        if article_column in r:
            return r.get(article_column, "")
        for candidate in ["article", "processed_body", "body", "text"]:
            if candidate in r and pd.notna(r.get(candidate)):
                return r.get(candidate, "")
        return ""

    def _classify_row(r):
        """Classify the full sample row by row"""
        res = classify_reactive(
            r.get("headline", ""), r.get("summary", ""), _article_text(r), threshold=threshold
        )
        return pd.Series({"is_reactive": res.is_reactive, "score": res.score})

    pred = sample_df.copy()
    if len(pred):
        pred[["is_reactive", "score"]] = pred.apply(_classify_row, axis=1)
    else:
        pred["is_reactive"] = pd.Series(dtype=int)
        pred["score"] = pd.Series(dtype=float)

    keep_cols = [
        c for c in ["article_id", "headline", "summary", article_column] if c in pred.columns
    ]
    pred_for_merge = pred[keep_cols + ["is_reactive", "score"]]
    merged = labelled_sample.merge(pred_for_merge, on="article_id", how="left")

    tp = int(((merged.hand_label == 1) & (merged.is_reactive == 1)).sum())
    fp = int(((merged.hand_label == 0) & (merged.is_reactive == 1)).sum())
    fn = int(((merged.hand_label == 1) & (merged.is_reactive == 0)).sum())
    tn = int(((merged.hand_label == 0) & (merged.is_reactive == 0)).sum())

    metrics = {
        "n": len(merged),
        "agreement": (tp + tn) / len(merged) if len(merged) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "base_rate_pred": merged.is_reactive.mean(),
        "base_rate_true": merged.hand_label.mean(),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

    return metrics
