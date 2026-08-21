"""Local instruct-LLM judge for coreference referent verification.

Asks, per sentence: does this mention denote the target company? Gating is per
sentence rather than per chain, since a chain can be right for some mentions and
wrong for others.

Fails closed. Answers are `yes` / `no` / `unsure` and accept_only() takes `yes`
alone; timeouts, malformed output and exceptions all discard. A GBNF grammar
admits only the three tokens, so malformed output is impossible by construction.

Verdicts are cached under MODEL_ID and PROMPT_VERSION, which is what makes a
feature set reproducible; temperature 0 only makes one process deterministic.

Ticker-agnostic: the target is a runtime parameter and the prompt names it from
COMPANIES[target]["names"][0]. It relies on world knowledge, so accuracy is not
uniform across tickers and nothing here detects when it has decayed.

    is_available()   whether llama-cpp and the model file are usable
    build_prompt()   the exact string shown to the model, for one context
    make_judge()     a judge callable for coref_eval.evaluate_judge()
    judge_corpus()   the batch entry point: chunked, resumable, merged back on
"""

from pathlib import Path
import time

from loguru import logger
import pandas as pd

from stock_predictor.config import COMPANIES, JUDGE_MODEL_PATH, PRIMARY_TICKER

# Bump on ANY change to the string the model sees: it is part of the cache key.
PROMPT_VERSION = "v3"

# Names the weights behind a verdict; a requantisation must change this.
MODEL_ID = "qwen2.5-7b-instruct-q4km"

# Rows per cache flush: the ceiling on what a crash costs during an hours-long pass.
JUDGE_CHUNK = 100

# Error classes in the prompt are named generically, never by sector: naming an
# industry would leak this corpus's target. A test guards against sector
# vocabulary. Kept in sync with coref_eval.JUDGE_ANSWERS by the test suite.
ANSWERS = ("yes", "no", "unsure")

# GBNF grammar admitting only the three answers.
ANSWER_GRAMMAR = 'root ::= "yes" | "no" | "unsure"'

# Corpus max prompt is 2,446 tokens, and llama.cpp raises rather than truncating
# above n_ctx.
DEFAULT_N_CTX = 4096

_SYSTEM_PROMPT = (
    "You are a careful financial-news analyst. You decide what a pronoun or "
    "definite description in a news sentence actually refers to, using the "
    "surrounding article as evidence. You answer with one word and nothing else."
)

# "the marked phrase" scopes the question to one mention, not the whole sentence.
# The `no` classes are named generically, which is what keeps this ticker-agnostic.
# The target's own products answer `yes`.
_INSTRUCTION = """Below is a news article extract. One phrase in the marked sentence is wrapped in **double asterisks**.

Question: does the marked phrase refer to {company}, or to something else?

Answer "yes" if the marked phrase refers to {company} itself, or to one of {company}'s own products.
Answer "no" if it refers to anything else -- a different company, a different company's products, a person, a fund or ETF that merely holds {company}, an industry or market, or a generic definition.
Answer "unsure" if the extract does not contain enough evidence to decide.

Answer with exactly one word: yes, no, or unsure."""

# No-span rows have no marked phrase, so they get a question matched to the weaker
# claim being made about them.
_INSTRUCTION_NO_SPAN = """Below is a news article extract. One sentence is marked with >.

Question: is the marked sentence making a statement about {company}?

Answer "yes" if the marked sentence is about {company} itself, or about one of {company}'s own products.
Answer "no" if it is about anything else -- a different company, a different company's products, a person, a fund or ETF that merely holds {company}, an industry or market, a generic definition, or promotional boilerplate.
Answer "unsure" if the extract does not contain enough evidence to decide.

Answer with exactly one word: yes, no, or unsure."""

_MODEL = None
_LOAD_FAILED = False


def company_name(target: str = PRIMARY_TICKER) -> str:
    """Display name for `target`, from the same config the aliases come from.

    Falls back to the ticker for a company with no entry: a prompt saying "TSLA"
    rather than "Tesla" is worse but still coherent.
    """
    entry = COMPANIES.get(target, {})
    names = entry.get("names") or []
    return names[0] if names else target


def build_prompt(ctx, target: str = PRIMARY_TICKER) -> str:
    """The exact string shown to the model for one JudgeContext.

    Separate from inference so the prompt can be diffed and tested without loading
    4.7GB of weights, and so a human can read what was asked when a verdict looks
    wrong.
    """
    company = company_name(target)
    instruction = _INSTRUCTION if ctx.has_span else _INSTRUCTION_NO_SPAN
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction.format(company=company)}\n\n"
        f"---\n{ctx.render()}\n---<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _load_model(model_path=None, n_ctx: int = DEFAULT_N_CTX, n_threads: int | None = None):
    """Load (once, then memoise) the GGUF model. Returns None on failure.

    _LOAD_FAILED latches so a missing model file warns once rather than on every row.
    """
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL

    path = Path(model_path or JUDGE_MODEL_PATH)
    try:
        if not path.exists():
            raise FileNotFoundError(f"no GGUF model at {path}")
        from llama_cpp import Llama

        logger.info(f"Loading judge model from {path} (CPU, n_ctx={n_ctx})")
        _MODEL = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            logits_all=False,
            verbose=False,
        )
        logger.info("Judge model loaded")
    except Exception as exc:  # noqa: BLE001 - any failure means "no judge"
        _LOAD_FAILED = True
        logger.warning(
            f"Judge backend unavailable ({type(exc).__name__}: {exc}); "
            "every row will be judged 'unsure', i.e. discarded"
        )
        _MODEL = None
    return _MODEL


def is_available(model_path=None) -> bool:
    """True when llama-cpp imported and the GGUF file loaded.

    A False must mean "discard everything", never "raise".
    """
    return _load_model(model_path) is not None


def make_judge(
    target: str = PRIMARY_TICKER,
    model_path=None,
    n_ctx: int = DEFAULT_N_CTX,
    n_threads: int | None = None,
    max_tokens: int = 4,
):
    """Build a judge callable for coref_eval.evaluate_judge().

    Returns a function of one JudgeContext returning `yes` / `no` / `unsure`. The model
    loads lazily on first call. Decoding is temperature 0 with the answer grammar
    attached; anything unexpected lands on `unsure` and is discarded.
    """

    def judge(ctx) -> str:
        model = _load_model(model_path, n_ctx=n_ctx, n_threads=n_threads)
        if model is None:
            return "unsure"
        try:
            out = model(
                build_prompt(ctx, target=target),
                max_tokens=max_tokens,
                temperature=0.0,
                grammar=_grammar(),
                stop=["<|im_end|>", "\n"],
            )
        except Exception as exc:  # noqa: BLE001 - fail closed, never kill the run
            # The judge itself must fail closed, not just the eval harness: a
            # prompt exceeding n_ctx once raised out of llama.cpp and killed a
            # corpus pass at 67%.
            logger.warning(
                f"Judge failed on {ctx.article_id}/{ctx.sent_idx} "
                f"({type(exc).__name__}: {exc}); returning 'unsure' (discard)"
            )
            return "unsure"
        answer = out["choices"][0]["text"].strip().lower()
        return answer if answer in ANSWERS else "unsure"

    return judge


_GRAMMAR = None


def _grammar():
    """Compile (once) the GBNF grammar restricting output to the three answers."""
    global _GRAMMAR
    if _GRAMMAR is None:
        from llama_cpp import LlamaGrammar

        _GRAMMAR = LlamaGrammar.from_string(ANSWER_GRAMMAR, verbose=False)
    return _GRAMMAR


def judge_population(sentences_df: pd.DataFrame) -> pd.Series:
    """Boolean mask of the sentences the judge is asked about.

        mentions_target & ~is_boilerplate & resolved_by_coref

    A sentence naming the company literally has no referent decision to verify.
    Boilerplate is excluded because needs_score() already drops it.

    Shared by the judging loop and the gate applying it, so the two cannot drift. A
    missing column reads as all-False.
    """

    def _flag(name: str) -> pd.Series:
        col = sentences_df.get(name, pd.Series(False, index=sentences_df.index))
        return col.fillna(False).astype(bool)

    return _flag("mentions_target") & ~_flag("is_boilerplate") & _flag("resolved_by_coref")


def judge_corpus(
    sentences_df: pd.DataFrame,
    articles: pd.DataFrame,
    target: str = PRIMARY_TICKER,
    judge=None,
    model_id: str = MODEL_ID,
    prompt_version: str = PROMPT_VERSION,
    chunk: int = JUDGE_CHUNK,
    cache_path=None,
) -> pd.DataFrame:
    """Judge every coref-resolved sentence in `sentences_df`, verdicts merged on.

    Returns a copy carrying two columns:

        judge_answer    `yes` / `no` / `unsure`, NA outside judge_population()
        judge_accepted  True when the row was never coref-resolved, or when its
                        verdict passes accept_only(). Fails closed.

    Chunked and resumable: verdicts flush every `chunk` rows to a cache keyed
    (article_id, sent_idx, target, model_id, prompt_version), merge-never-replace, so
    a re-run resumes rather than restarting. A fully cached corpus never loads the
    weights.

    `judge` is injectable so the loop can run without weights. `articles` supplies each
    context window's headline; both frames are passed in, keeping this a transform.
    """
    from stock_predictor.text.coref_eval import (
        CACHE_COLUMNS,
        accept_only,
        build_context,
        load_judge_cache,
        save_judge_cache,
    )

    cache_kwargs = {} if cache_path is None else {"path": cache_path}

    out = sentences_df.copy()
    pop = out[judge_population(out)]
    spans = pop["mention_char_start"] if "mention_char_start" in pop.columns else None
    n_span = int(spans.notna().sum()) if spans is not None else 0
    logger.info(
        f"judge_corpus: population {len(pop)} coref sentences "
        f"({n_span} span, {len(pop) - n_span} no-span) across "
        f"{pop['article_id'].nunique() if len(pop) else 0} articles"
    )

    cache = load_judge_cache(**cache_kwargs)
    mine = cache[(cache["model_id"] == model_id) & (cache["prompt_version"] == prompt_version)]
    answered = set(zip(mine["article_id"], mine["sent_idx"]))
    keys = pd.Series(list(zip(pop["article_id"], pop["sent_idx"])), index=pop.index, dtype=object)
    todo = pop[~keys.isin(answered)] if len(pop) else pop
    logger.info(
        f"judge_corpus: {len(answered)} verdicts cached under "
        f"{model_id}/{prompt_version}; {len(todo)} to judge"
    )

    if len(todo):
        head_col = "headline" if "headline" in articles.columns else "title"
        headlines = articles.set_index("article_id")[head_col].to_dict()
        by_article = {aid: g.sort_values("sent_idx") for aid, g in out.groupby("article_id")}
        if judge is None:
            judge = make_judge(target=target)

        fresh, start = [], time.time()
        for i, row in enumerate(todo.itertuples(index=False), start=1):
            ctx = build_context(
                article_id=row.article_id,
                sent_idx=row.sent_idx,
                sentences=by_article[row.article_id],
                headlines=headlines,
                target=target,
                mention_char_start=getattr(row, "mention_char_start", None),
                mention_char_end=getattr(row, "mention_char_end", None),
            )
            fresh.append(
                {
                    "article_id": int(row.article_id),
                    "sent_idx": int(row.sent_idx),
                    "target": target,
                    "model_id": model_id,
                    "prompt_version": prompt_version,
                    "answer": judge(ctx),
                }
            )
            if i % chunk == 0 or i == len(todo):
                # Merge, never replace -- `fresh` holds only this run's verdicts,
                # and the cache is re-read so a longer-running sibling is not
                # clobbered by this save.
                save_judge_cache(
                    pd.concat(
                        [
                            load_judge_cache(**cache_kwargs),
                            pd.DataFrame(fresh, columns=CACHE_COLUMNS),
                        ],
                        ignore_index=True,
                    ),
                    **cache_kwargs,
                )
                fresh = []
                rate = (time.time() - start) / i
                logger.info(
                    f"judge_corpus: {i}/{len(todo)} ({i / len(todo):.1%}) "
                    f"{rate:.2f}s/row ~{(len(todo) - i) * rate / 3600:.1f}h left"
                )

    cache = load_judge_cache(**cache_kwargs)
    mine = cache[(cache["model_id"] == model_id) & (cache["prompt_version"] == prompt_version)]
    out = out.merge(
        mine[["article_id", "sent_idx", "answer"]].rename(columns={"answer": "judge_answer"}),
        on=["article_id", "sent_idx"],
        how="left",
    )
    # Recomputed after the merge: a verdict cached for a row no longer in the
    # population must not resurrect.
    in_population = judge_population(out)
    out.loc[~in_population, "judge_answer"] = pd.NA

    # Only rows the judge was asked about can be rejected: outside the population
    # there was no question, so a surface match passes.
    out["judge_accepted"] = ~in_population | out["judge_answer"].map(accept_only).astype(bool)

    verdicts = out["judge_answer"].value_counts().to_dict()
    logger.info(
        f"judge_corpus: {int(out['judge_answer'].notna().sum())} rows carry a verdict "
        f"({verdicts}); {int((~out['judge_accepted']).sum())} rows dropped by the gate"
    )
    return out
