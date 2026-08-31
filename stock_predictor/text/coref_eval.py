"""Measuring a coreference referent-verification judge against hand labels.

A judge is any callable reading one sentence in context and answering `yes` / `no`
/ `unsure`. accept_only() is the single definition of the accept rule: `yes` alone,
everything else including errors and timeouts discards.

The headline metric is accept-precision, of the rows a judge accepts what fraction
were about the target, since rejected rows never reach the model. Every metric is
reported against an accept-everything baseline, and split by has_span into `span`,
`no-span` and `all`, because the two sub-populations differ sharply.

Rows flagged `borderline` are counted as `target` by convention, and figures are
reported with and without them.

build_context() is the single definition of what a judge reads, matching what the
human auditor read; verify_contexts_match() checks it still reproduces the text
stored in the eval parquet.

Verdicts are keyed (article_id, sent_idx, target, model_id, prompt_version) and
persisted, so a reworded prompt cannot reuse old answers. Saves MERGE with the
cache loaded at call start.

    load_eval_set()          the hand-labelled frame, validated
    build_context()          one JudgeContext for an (article_id, sent_idx)
    build_contexts()         contexts aligned 1:1 with an eval frame
    verify_contexts_match()  assert the builder reproduces the eval text
    accept_only()            the yes/no/unsure -> accept/discard rule
    evaluate_judge()         metrics vs the labels, split and baselined
    load_judge_cache() / save_judge_cache()
"""

from dataclasses import dataclass

from loguru import logger
import numpy as np
import pandas as pd

from stock_predictor.config import (
    COREF_EVAL_PATH,
    COREF_JUDGE_CACHE_PATH,
    EVAL_CONTEXT_FOLLOWING,
    EVAL_CONTEXT_PRECEDING,
    PRIMARY_TICKER,
)

# Columns the labelled eval set must carry for anything here to mean anything.
# `verdict` is the ground truth: 'target' = the coref resolution was correct,
# 'other' = it resolved to something else and the row is an error.
EVAL_COLUMNS = [
    "row_id",
    "article_id",
    "sent_idx",
    "text",
    "has_span",
    "verdict",
    "borderline",
]

VERDICT_TARGET = "target"
VERDICT_OTHER = "other"

# Anything outside this set is treated as `unsure` rather than raising: a
# malformed answer on row 4,000 of a 3-hour run must not lose the run.
JUDGE_ANSWERS = ("yes", "no", "unsure")

CACHE_COLUMNS = [
    "article_id",
    "sent_idx",
    "target",
    "model_id",
    "prompt_version",
    "answer",
]

# 'all' is reported last and never alone; see the module docstring on has_span.
POPULATIONS = ("span", "no-span", "all")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n.

    Wilson rather than the normal approximation: these samples are small and the
    rates sit near the ends of the range, where the normal interval runs past 0 or 1.
    Identical to notebook 2.1's definition so the numbers are comparable.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


@dataclass(frozen=True)
class JudgeContext:
    """Everything a judge is allowed to read about one sentence.

    Frozen so one judge cannot mutate what the next one sees.

    `mention_char_start` / `mention_char_end` are offsets into `sentence`, None for
    no-span rows.
    """

    article_id: int
    sent_idx: int
    target: str
    headline: str
    preceding: tuple[str, ...]
    sentence: str
    following: tuple[str, ...]
    mention_char_start: int | None = None
    mention_char_end: int | None = None

    @property
    def has_span(self) -> bool:
        return self.mention_char_start is not None and self.mention_char_end is not None

    @property
    def mention(self) -> str | None:
        """The literal text the pipeline would overwrite, or None if no span."""
        if not self.has_span:
            return None
        return self.sentence[self.mention_char_start : self.mention_char_end]

    def marked_sentence(self, marker: str = "**") -> str:
        """The sentence with the mention delimited, or unchanged if no span.

        The human labeller was shown which mention was under question; a judge
        asked "does the mention denote the target" without being told WHICH
        mention is answering a different, harder question.
        """
        if not self.has_span:
            return self.sentence
        start, end = self.mention_char_start, self.mention_char_end
        return f"{self.sentence[:start]}{marker}{self.sentence[start:end]}{marker}{self.sentence[end:]}"

    def render(self, marker: str = "**") -> str:
        """The context as a judge (or a human) reads it: headline, then flow.

        Kept here rather than in a prompt template so that every judge and every
        labelling sheet agree on what "the context" is by construction instead
        of by convention.
        """
        lines = [f"HEADLINE: {self.headline}", ""]
        lines.extend(f"  {s}" for s in self.preceding)
        lines.append(f"> {self.marked_sentence(marker)}")
        lines.extend(f"  {s}" for s in self.following)
        return "\n".join(lines)


def load_eval_set(path=COREF_EVAL_PATH) -> pd.DataFrame:
    """Load the hand-labelled eval set, validating the columns it must have.

    Raises rather than degrading, unlike the coref cluster cache: a missing or
    schema-drifted eval set is not a slow path, it is the ground truth being
    absent, and every number computed downstream would be meaningless. Failing
    loudly here is the whole point of the module.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No labelled eval set at {path}. Nothing can be measured without it; "
            "see notebook 2.1 for how it was built."
        )
    df = pd.read_parquet(path)
    missing = set(EVAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Eval set at {path} is missing columns: {sorted(missing)}")

    bad = set(df["verdict"].unique()) - {VERDICT_TARGET, VERDICT_OTHER}
    if bad:
        raise ValueError(f"Eval set has unexpected verdict values: {sorted(bad)}")

    logger.info(
        f"Loaded {len(df)} labelled rows from {path} "
        f"({int(df['has_span'].sum())} span, {int((~df['has_span']).sum())} no-span, "
        f"{int((df['verdict'] == VERDICT_OTHER).sum())} errors)"
    )
    return df


def build_context(
    article_id: int,
    sent_idx: int,
    sentences: pd.DataFrame,
    headlines: dict,
    target: str = PRIMARY_TICKER,
    mention_char_start=None,
    mention_char_end=None,
    n_preceding: int = EVAL_CONTEXT_PRECEDING,
    n_following: int = EVAL_CONTEXT_FOLLOWING,
) -> JudgeContext:
    """Build the context window for one sentence.

    `headlines` maps article_id -> headline. Both frames are passed in rather than
    loaded here, so evaluating 3,617 rows reads the parquet once.

    The window is the labelling convention, not a tunable: headline, 4 preceding
    sentences, the sentence, 1 following. Deviating means the judge is not solving
    the task the labels describe.
    """
    article = sentences[sentences["article_id"] == article_id].sort_values("sent_idx")
    row = article[article["sent_idx"] == sent_idx]
    if row.empty:
        raise KeyError(f"No sentence {sent_idx} in article {article_id}")

    preceding = article[
        (article["sent_idx"] < sent_idx) & (article["sent_idx"] >= sent_idx - n_preceding)
    ]
    following = article[
        (article["sent_idx"] > sent_idx) & (article["sent_idx"] <= sent_idx + n_following)
    ]

    def _clean(value):
        return None if value is None or pd.isna(value) else int(value)

    return JudgeContext(
        article_id=int(article_id),
        sent_idx=int(sent_idx),
        target=target,
        headline=str(headlines.get(article_id, "")),
        preceding=tuple(preceding["text"].astype(str)),
        sentence=str(row["text"].iloc[0]),
        following=tuple(following["text"].astype(str)),
        mention_char_start=_clean(mention_char_start),
        mention_char_end=_clean(mention_char_end),
    )


def build_contexts(
    eval_df: pd.DataFrame,
    sentences: pd.DataFrame,
    articles: pd.DataFrame,
    target: str = PRIMARY_TICKER,
    **kwargs,
) -> list[JudgeContext]:
    """One JudgeContext per eval row, in the eval frame's own order.

    Order matters: `evaluate_judge()` zips these against the frame positionally,
    so a reordering here would silently score every judgement against the wrong
    label -- the kind of bug that produces a plausible number and no error.
    """
    head_col = "headline" if "headline" in articles.columns else "title"
    headlines = articles.set_index("article_id")[head_col].to_dict()

    has_spans = "mention_char_start" in eval_df.columns
    return [
        build_context(
            article_id=row.article_id,
            sent_idx=row.sent_idx,
            sentences=sentences,
            headlines=headlines,
            target=target,
            mention_char_start=getattr(row, "mention_char_start", None) if has_spans else None,
            mention_char_end=getattr(row, "mention_char_end", None) if has_spans else None,
            **kwargs,
        )
        for row in eval_df.itertuples(index=False)
    ]


def verify_contexts_match(eval_df: pd.DataFrame, contexts: list[JudgeContext]) -> None:
    """Assert the rebuilt context reproduces the sentence text in the eval set.

    A regenerated sentences.parquet or a changed splitter would have the judge
    reading a different sentence from the one labelled, while every metric still
    computed cleanly. Called by evaluate_judge() before any judging.
    """
    mismatches = [
        (row.row_id, ctx.sentence, row.text)
        for row, ctx in zip(eval_df.itertuples(index=False), contexts)
        if str(ctx.sentence) != str(row.text)
    ]
    if mismatches:
        row_id, rebuilt, stored = mismatches[0]
        raise ValueError(
            f"{len(mismatches)} of {len(contexts)} contexts do not match the labelled text. "
            f"First: {row_id}\n  rebuilt: {rebuilt!r}\n  stored : {stored!r}\n"
            "The corpus and the eval set have drifted apart; re-derive one or the other."
        )


def accept_only(answer: str) -> bool:
    """The accept/discard rule: accept iff the judge said exactly `yes`.

    Fail-closed, in one place, so no caller can quietly decide that `unsure` is
    good enough. Anything unrecognised -- an empty string, a stack trace, a
    model that answered in a sentence -- is a discard, not an exception.
    """
    return isinstance(answer, str) and answer.strip().lower() == "yes"


def load_judge_cache(path=COREF_JUDGE_CACHE_PATH) -> pd.DataFrame:
    """Load the on-disk judge verdict cache; empty frame when absent.

    Never raises. An unreadable cache costs a re-run, which is expensive but
    correct; refusing to start because of it would be worse.
    """
    if not path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    try:
        df = pd.read_parquet(path)
        missing = set(CACHE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {missing}")
        return df[CACHE_COLUMNS]
    except Exception as exc:  # noqa: BLE001 - unreadable cache == no cache
        logger.warning(
            f"Judge cache at {path} is unreadable ({type(exc).__name__}: {exc}); "
            "treating it as empty"
        )
        return pd.DataFrame(columns=CACHE_COLUMNS)


def save_judge_cache(cache_df: pd.DataFrame, path=COREF_JUDGE_CACHE_PATH) -> None:
    """Persist verdicts, deduped on the full key (keep last).

    Callers must pass the MERGE of the cache they loaded and the rows they
    produced -- see the module docstring on why replacing is a bug and not a
    simplification.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    key = ["article_id", "sent_idx", "target", "model_id", "prompt_version"]
    deduped = cache_df.drop_duplicates(subset=key, keep="last")[CACHE_COLUMNS]
    deduped.to_parquet(path, index=False)
    logger.info(f"Saved judge cache with {len(deduped)} verdicts to {path}")


def _metrics(truth_is_target: pd.Series, accepted: pd.Series) -> dict:
    """Metrics for one population under one accept/discard decision.

    `truth_is_target` and `accepted` are aligned boolean Series.

    accept_precision is the headline: of what survives, how much is right. The rest
    stop it being read alone, since accepting a single row scores 100%.
    """
    n = len(truth_is_target)
    n_errors = int((~truth_is_target).sum())
    n_correct = int(truth_is_target.sum())
    n_accepted = int(accepted.sum())

    accepted_correct = int((accepted & truth_is_target).sum())
    errors_caught = int((~accepted & ~truth_is_target).sum())
    correct_lost = int((~accepted & truth_is_target).sum())

    precision = accepted_correct / n_accepted if n_accepted else float("nan")
    lo, hi = wilson_ci(accepted_correct, n_accepted) if n_accepted else (float("nan"),) * 2

    return {
        "n": n,
        "n_errors": n_errors,
        "baseline_precision": n_correct / n if n else float("nan"),
        "n_accepted": n_accepted,
        "accept_rate": n_accepted / n if n else float("nan"),
        "accept_precision": precision,
        "accept_precision_lo": lo,
        "accept_precision_hi": hi,
        "error_recall": errors_caught / n_errors if n_errors else float("nan"),
        "correct_lost": correct_lost,
        "correct_lost_rate": correct_lost / n_correct if n_correct else float("nan"),
    }


def evaluate_judge(
    judge,
    eval_df: pd.DataFrame,
    contexts: list[JudgeContext] | None = None,
    sentences: pd.DataFrame | None = None,
    articles: pd.DataFrame | None = None,
    target: str = PRIMARY_TICKER,
    model_id: str = "unknown",
    prompt_version: str = "v0",
    cache_path=COREF_JUDGE_CACHE_PATH,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Score a judge against the hand labels. Returns one row per slice.

    `judge` takes a JudgeContext and returns `yes` / `no` / `unsure`. It may raise:
    exceptions are caught, logged once, and counted as `unsure`, so a judge dying on
    3% of rows shows up as a lower accept rate rather than a crashed evaluation.

    Either pass `contexts`, or `sentences` + `articles` to have them built here.
    Contexts are verified against the labelled text first.

    Carries, per population and per `borderline` treatment, the metrics from
    _metrics() plus an accept-everything baseline row. Read the baseline first.
    """
    if contexts is None:
        if sentences is None or articles is None:
            raise ValueError("Pass either contexts= or both sentences= and articles=")
        contexts = build_contexts(eval_df, sentences, articles, target=target)
    if len(contexts) != len(eval_df):
        raise ValueError(f"{len(contexts)} contexts for {len(eval_df)} eval rows")
    verify_contexts_match(eval_df, contexts)

    cache = load_judge_cache(cache_path) if use_cache else pd.DataFrame(columns=CACHE_COLUMNS)
    cached = {
        (r.article_id, r.sent_idx, r.target, r.model_id, r.prompt_version): r.answer
        for r in cache.itertuples(index=False)
    }

    answers, fresh, n_hits, n_errors = [], [], 0, 0
    for ctx in contexts:
        key = (ctx.article_id, ctx.sent_idx, target, model_id, prompt_version)
        if key in cached:
            answers.append(cached[key])
            n_hits += 1
            continue
        try:
            answer = judge(ctx)
        except Exception as exc:  # noqa: BLE001 - fail closed, never lose the run
            if n_errors == 0:
                logger.warning(
                    f"Judge raised on {ctx.article_id}/{ctx.sent_idx} "
                    f"({type(exc).__name__}: {exc}); counting as 'unsure'. "
                    "Further failures in this run are not logged individually."
                )
            n_errors += 1
            answer = "unsure"
        if answer not in JUDGE_ANSWERS:
            answer = "unsure"
        answers.append(answer)
        fresh.append(
            {
                "article_id": ctx.article_id,
                "sent_idx": ctx.sent_idx,
                "target": target,
                "model_id": model_id,
                "prompt_version": prompt_version,
                "answer": answer,
            }
        )

    logger.info(
        f"evaluate_judge: {len(contexts)} rows ({n_hits} cache hits, "
        f"{len(fresh)} judged, {n_errors} judge failures counted as 'unsure')"
    )

    if use_cache and fresh:
        # Merge, never replace: `fresh` is only this run's misses.
        save_judge_cache(
            pd.concat([cache, pd.DataFrame(fresh, columns=CACHE_COLUMNS)], ignore_index=True),
            path=cache_path,
        )

    scored = eval_df.copy()
    scored["answer"] = answers
    scored["accepted"] = [accept_only(a) for a in answers]
    scored["is_target"] = scored["verdict"] == VERDICT_TARGET

    rows = []
    for borderline_mode in ("included", "excluded"):
        frame = scored if borderline_mode == "included" else scored[~scored["borderline"]]
        for population in POPULATIONS:
            if population == "span":
                sub = frame[frame["has_span"]]
            elif population == "no-span":
                sub = frame[~frame["has_span"]]
            else:
                sub = frame
            if sub.empty:
                continue
            for name, accepted in (
                ("baseline (accept all)", pd.Series(True, index=sub.index)),
                (model_id, sub["accepted"]),
            ):
                rows.append(
                    {
                        "judge": name,
                        "prompt_version": prompt_version if name == model_id else "-",
                        "population": population,
                        "borderline": borderline_mode,
                        **_metrics(sub["is_target"], accepted),
                    }
                )
    return pd.DataFrame(rows)
