"""Measuring a coreference referent-verification judge against hand labels.

WHY THIS EXISTS, AND WHY IT EXISTS BEFORE THE JUDGE DOES. Two attempts have
already been made to suppress wrong coref resolutions with hand-reasoned rules.
Both were built on argument, shipped without being measured against labels, and
both turned out to discard ~90% of the WRONG rows they were aimed at only by
also discarding most of the right ones. The third attempt (a local instruct-LLM
judge) is the expensive one, and the only thing that separates it from attempts
1 and 2 is that this module exists first. Nothing that claims to verify a
referent goes near the pipeline until it has a row in `evaluate_judge()`'s
output.

WHAT A "JUDGE" IS HERE. Any callable that reads one sentence in context and
answers "does the anaphor in this sentence denote the target company?" with
exactly one of `yes` / `no` / `unsure`. That three-way contract is deliberate
and is not a stylistic choice: the project's posture is that losing a sentence
is acceptable and corrupting one is not, so the pipeline **accepts on `yes` and
discards on anything else** — including on the judge erroring out or timing out.
`unsure` therefore is not a third outcome to be handled later, it is a discard
that happens to be honest about why. `accept_only()` implements exactly this
rule and is the only place it is written down.

THE METRIC THAT MATTERS IS ACCEPT-PRECISION, not accuracy and not F1. Of the
rows a judge accepts, what fraction were genuinely about the target? That is the
quantity the downstream sentiment model is exposed to, because rejected rows
simply never reach it. Recall is reported but is explicitly negotiable — a judge
that discards half the corpus and is right about everything it keeps is a good
outcome here, and a judge with beautiful F1 and 85% accept-precision is not.
Every metric is also reported against an ACCEPT-EVERYTHING BASELINE, i.e. what
the pipeline does today, because "the judge scores 89%" is meaningless until you
know that doing nothing scores 89% too.

ALWAYS SPLIT BY has_span. The two coref sub-populations are 89.0% and 56.5%
correct respectively (n=100 / n=170, notebook 2.7) — different enough that a
pooled number hides the result rather than summarising it. A judge could improve
the pooled figure while making the span channel worse, and a pooled report would
not show it. Every metric therefore comes back three times: `span`, `no-span`,
and `all`.

BORDERLINE ROWS BOTH WAYS. 37 of the 270 labelled rows are flagged `borderline`
(defensible either way — mostly plural/dual-entity sentences: "both firms",
"Musk's companies"). They are counted as `target` by the labelling convention,
but any headline figure computed from them is reported twice, with and without,
because a judge that only looks good when the ambiguous rows are counted its way
has not been shown to be good. That 37 is worth internalising: **36 of the 37
are no-span rows**, so excluding them shrinks the no-span sample from 170 to 134
while leaving the span sample essentially untouched (100 -> 99). The two
borderline treatments are therefore not a cosmetic sensitivity check on the
no-span channel — they are two materially different samples of it.

THE CONTEXT BUILDER IS SHARED ON PURPOSE. The human auditor who produced the
labels read the headline, 4 preceding sentences, the sentence itself and 1
following sentence. A judge given less than that is not being measured against
the labels, it is being measured against a harder task than the human solved —
and the consultation that motivated this whole stage put it bluntly: *a method
that reads less than the human auditor needed cannot match the auditor*. So
`build_context()` is the single definition of "what gets read", used both to
regenerate labelling sheets and to feed judges, and `verify_context_matches()`
checks it still reproduces the exact sentence text stored in the eval parquet.

THE CACHE IS WHAT MAKES A JUDGE REPRODUCIBLE, not temperature 0. A local LLM at
temperature 0 is reproducible in principle and not in practice — a driver
update, a quantisation change or a llama.cpp bump moves outputs. Verdicts are
therefore keyed on (article_id, sent_idx, target, model_id, prompt_version) and
persisted, exactly like the three existing caches in data/interim/. The key
includes prompt_version specifically so that editing the prompt does NOT
silently reuse verdicts formed under the old one — that is the failure this
scheme exists to prevent. Saves MERGE with the cache loaded at the start of the
call and never replace it; sentiment.py had a truncation bug from replacing, and
the same shape of bug here would throw away hours of CPU inference.

API:
    load_eval_set()          -> the hand-labelled frame, validated
    build_context()          -> one JudgeContext for an (article_id, sent_idx)
    build_contexts()         -> contexts aligned 1:1 with an eval frame
    verify_contexts_match()  -> assert the builder reproduces the eval text
    accept_only()            -> the yes/no/unsure -> accept/discard rule
    evaluate_judge()         -> metrics vs the labels, split and baselined
    load_judge_cache() / save_judge_cache()
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

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

# The judge contract. Anything not in this set is a protocol violation by the
# judge and is treated as `unsure` (i.e. discarded) rather than raising -- a
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

# Population labels used in evaluate_judge()'s output. 'all' is reported last
# and deliberately never alone -- see the module docstring on has_span.
POPULATIONS = ("span", "no-span", "all")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n.

    Wilson rather than the normal approximation because these samples are small
    (n=100 span, n=170 no-span, and much smaller once sliced) and rates sit near
    the ends of the range where the normal interval runs off past 0 or 1 and
    stops meaning anything. Same definition notebook 2.7 uses -- kept identical
    so a number computed here and a number computed there are comparable.
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

    Frozen because a judge must not be able to mutate what the next judge sees;
    an accidental in-place edit inside one judge implementation would otherwise
    silently change the task for everything evaluated after it in the same run.

    `anaphor_char_start`/`anaphor_char_end` are offsets into `sentence` and are
    None for no-span rows -- those were tagged without any substitutable mention
    being found, which is exactly why they are a separate population.
    """

    article_id: int
    sent_idx: int
    target: str
    headline: str
    preceding: tuple[str, ...]
    sentence: str
    following: tuple[str, ...]
    anaphor_char_start: int | None = None
    anaphor_char_end: int | None = None

    @property
    def has_span(self) -> bool:
        return self.anaphor_char_start is not None and self.anaphor_char_end is not None

    @property
    def anaphor(self) -> str | None:
        """The literal text the pipeline would overwrite, or None if no span."""
        if not self.has_span:
            return None
        return self.sentence[self.anaphor_char_start : self.anaphor_char_end]

    def marked_sentence(self, marker: str = "**") -> str:
        """The sentence with the anaphor delimited, or unchanged if no span.

        The human labeller was shown which mention was under question; a judge
        asked "does the anaphor denote the target" without being told WHICH
        anaphor is answering a different, harder question.
        """
        if not self.has_span:
            return self.sentence
        start, end = self.anaphor_char_start, self.anaphor_char_end
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
            "see notebook 2.7 for how it was built."
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
    anaphor_char_start=None,
    anaphor_char_end=None,
    n_preceding: int = EVAL_CONTEXT_PRECEDING,
    n_following: int = EVAL_CONTEXT_FOLLOWING,
) -> JudgeContext:
    """Build the context window for one sentence.

    `sentences` is the sentence table (or any subset containing this article);
    `headlines` maps article_id -> headline. Both are passed in rather than
    loaded here so that a caller evaluating 3,617 rows reads the parquet once
    instead of 3,617 times.

    The window is the labelling convention, not a tunable: headline + 4
    preceding + the sentence + 1 following. Deviating from it means the judge is
    not solving the task the labels describe.
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
        anaphor_char_start=_clean(anaphor_char_start),
        anaphor_char_end=_clean(anaphor_char_end),
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

    has_spans = "anaphor_char_start" in eval_df.columns
    return [
        build_context(
            article_id=row.article_id,
            sent_idx=row.sent_idx,
            sentences=sentences,
            headlines=headlines,
            target=target,
            anaphor_char_start=getattr(row, "anaphor_char_start", None) if has_spans else None,
            anaphor_char_end=getattr(row, "anaphor_char_end", None) if has_spans else None,
            **kwargs,
        )
        for row in eval_df.itertuples(index=False)
    ]


def verify_contexts_match(eval_df: pd.DataFrame, contexts: list[JudgeContext]) -> None:
    """Assert the rebuilt context reproduces the sentence text stored in the eval set.

    A silent drift between the corpus and the labels -- a regenerated
    sentences.parquet, a changed sentence splitter -- would mean the judge is
    reading a different sentence from the one the human labelled, while every
    metric still computes cleanly. This is the check that turns that into a
    loud failure. Called by evaluate_judge() before any judging happens.
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

    accept_precision is the headline: of what survives, how much is right. The
    remaining fields exist to stop it being read in isolation -- a judge can
    reach 100% accept-precision by accepting one row, which `accept_rate` and
    `correct_lost` make immediately obvious.
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

    `judge` is any callable taking a JudgeContext and returning `yes`/`no`/
    `unsure`. It may raise: an exception is caught, logged once per run, and
    counted as `unsure` -- i.e. a discard -- because the pipeline's rule is to
    fail closed, and a judge that dies on 3% of rows should show up as a lower
    accept rate, not as a crashed evaluation.

    Either pass `contexts` (already built) or `sentences` + `articles` and they
    will be built here. Contexts are verified against the labelled text before
    anything is judged.

    The returned frame carries, for each population in POPULATIONS and each
    `borderline` treatment ('included' / 'excluded'), the metrics from
    `_metrics()` plus a BASELINE row per slice for accept-everything. Read the
    baseline first: a judge is only interesting where it beats it.
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
