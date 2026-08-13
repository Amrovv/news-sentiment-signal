import re
from dataclasses import dataclass
from stock_predictor.config import INTERIM_DATA_DIR

import pandas as pd

#Verbs which indicate price movement discussion
MOVE_VERBS = (
    r"(rises?|falls?|slides?|slips?|surges?|jumps?|climbs?|tumbles?|sinks?|sank|sunk|"
    r"gains?|drops?|plunges?|soars?|rallies?|retreats?|declines?|advances?|"
    r"rose|fell|slid|slipped|surged|jumped|climbed|tumbled|sank|"
    r"gained|dropped|plunged|soared|rallied|retreated|declined|advanced)"
)

#Strong: near-certain price-commentary signals
STRONG_PATTERNS = [
    rf"\bshares?\s+(?:\w+\s+){{0,2}}{MOVE_VERBS}",
    rf"\bstock\s+(?:\w+\s+){{0,2}}{MOVE_VERBS}",
    r"\bstock\s+(closed|opened|traded|finished)\b",
    r"\b(up|down|off)\s+\d+(\.\d+)?\s*%",
    r"\b\d+(\.\d+)?\s*%\s+(higher|lower|gain|loss|drop|jump)",
    r"\bextend(s|ed|ing)?\s+(gains|losses|slide|rally)\b",
    r"\bhit(s|ting)?\s+(a\s+)?(new\s+)?(record|all-time)\s+(high|low)\b",
    r"\b(52|fifty-two)[- ]week\s+(high|low)\b",
]

#Weak: suggestive but common in analysis pieces too
WEAK_PATTERNS = [
    r"\b(rally|selloff|sell-off|slump|surge|plunge)\b",
    r"\bhere'?s\s+why\b",
    r"\binvestors?\s+(weighed|reacted|shrugged|cheered|punished)\b",
    r"\b(outperform|underperform)(ed|ing)?\s+the\s+(market|s&p|nasdaq)\b",
    r"\b(premarket|pre-market|after[- ]hours)\s+(trading|movers?)\b",
    r"\bmoving\s+(the\s+)?(market|stock)s?\b",
    r"\b(top|biggest)\s+(gainers?|losers?|movers?)\b",
    r"\bwhat'?s\s+(driving|behind)\b",
]

_STRONG = [re.compile(p, re.IGNORECASE) for p in STRONG_PATTERNS]
_WEAK = [re.compile(p, re.IGNORECASE) for p in WEAK_PATTERNS]

@dataclass
class ReactiveResult:
    is_reactive: int          #0/1 reactive flag
    score: float              #Score based on the hits
    hits: list[str]           #Which patterns fired (for debugging/tuning)


def check_hits(text: str, patterns) -> list[str]:
    """
    Returns a list of hits from the given text

    text: string which contains summary/headline to be checked
    patterns: list of patterns that we are looking for
    """

    if not text:
        return []
    return [p.search(text).group() for p in patterns if p.search(text)]


def classify_reactive(headline: str, summary: str = "", article: str="", 
                      threshold: float = 1.0) -> ReactiveResult:
    """
    Classify a whole article.

    Weights: headline strong=1.0, headline weak=0.5,
             summary strong=0.5, summary weak=0.25,
    Default threshold: 1.0 -> a single strong headline hit is enough.
    """
    #Pull hits
    h_strong = check_hits(headline, _STRONG)
    h_weak = check_hits(headline, _WEAK)
    s_strong = check_hits(summary, _STRONG)
    s_weak = check_hits(summary, _WEAK)

    #Score the hits accordingly
    score = (
        1.00 * len(h_strong)
        + 0.50 * len(h_weak)
        + 0.50 * len(s_strong)
        + 0.25 * len(s_weak)
    )

    #H = headline, S = summary
    #! = strong, ? = weak
    matched = (
        [f"H! {p}" for p in h_strong]
        + [f"H? {p}" for p in h_weak]
        + [f"S! {p}" for p in s_strong]
        + [f"S? {p}" for p in s_weak]
    )
    return ReactiveResult(int(score >= threshold), score, matched)

def make_labelling_sample(article_df: pd.DataFrame, seed: int, 
                          n: int = 100):
    """
    Draw a random sample and write a CSV with a blank `true_label` column.
    Fill it in by hand (1 = reactive, 0 = not), then feed it to evaluate().

    Deliberately omits the predicted flag so your hand labels stay unbiased.
    """
    path = INTERIM_DATA_DIR / "reactive_sample.csv"
    #Generate random sample
    sample = article_df.sample(n=min(n, len(article_df)), random_state=seed)[
        ["article_id", "headline", "summary"]
    ].copy()
    sample["hand_label"] = ""
    sample.to_csv(path, index=False)
    print(f"Wrote {len(sample)} rows to {path} — fill in true_label, then run evaluate().")
    return sample


def evaluate_sample(labelled_path: str, sample_df: pd.DataFrame, threshold: float = 1.0) -> dict:
    """
    Evaluate the hand labelled articles against our algorithm.
    """
    #Pull the labelled sample file
    labelled_sample = pd.read_csv(labelled_path)
    labelled_sample = labelled_sample[labelled_sample["hand_label"].notna() & 
                                      (labelled_sample["hand_label"] != "")]
    labelled_sample["hand_label"] = labelled_sample["hand_label"].astype(int)

    def _classify_row(r):
        """
        Classify each row from the sample individually.
        """
        res = classify_reactive(r.get("headline", ""), r.get("summary", ""), r.get("article", ""), threshold=threshold)
        return pd.Series({"is_reactive": res.is_reactive, "score": res.score})

    #Classify each row
    pred = sample_df.copy()
    if len(pred):
        pred[["is_reactive", "score"]] = pred.apply(_classify_row, axis=1)
    else:
        pred["is_reactive"] = pd.Series(dtype=int)
        pred["score"] = pd.Series(dtype=float)

    #Merge the hand labels with algorithm labels
    pred = pred[["article_id", "is_reactive", "score"]]
    merged = labelled_sample.merge(pred, on="article_id", how="left")

    #Check for true positives/negatives and false positives/negatives
    tp = int(((merged.hand_label == 1) & (merged.is_reactive == 1)).sum())
    fp = int(((merged.hand_label == 0) & (merged.is_reactive == 1)).sum())
    fn = int(((merged.hand_label == 1) & (merged.is_reactive == 0)).sum())
    tn = int(((merged.hand_label == 0) & (merged.is_reactive == 0)).sum())

    #Calculate metrics
    metrics = {
        "n": len(merged),
        "agreement": (tp + tn) / len(merged) if len(merged) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "base_rate_pred": merged.is_reactive.mean(),
        "base_rate_true": merged.hand_label.mean(),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

    #Output the metrics and the articles the algorithm got wrong for tuning
    errors = merged[merged.hand_label != merged.is_reactive]
    print(f"Agreement {metrics['agreement']:.2%} on n={metrics['n']} "
          f"(precision {metrics['precision']:.2%}, recall {metrics['recall']:.2%})")
    print(f"\n{len(errors)} disagreements:")
    for _, r in errors.head(20).iterrows():
        print(f"  [hand label={r.hand_label}, algo label={r.is_reactive}] {r.headline[:90]}")

    return metrics