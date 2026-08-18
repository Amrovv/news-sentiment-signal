"""A local instruct-LLM judge for coreference referent verification.

WHAT THIS ANSWERS, AND WHY IT IS A DIFFERENT QUESTION FROM THE TWO THAT FAILED.
Attempts 1 and 2 both asked an ANOMALY-DETECTION question -- "is some other
company mentioned near this chain?" -- and both failed for the same structural
reason: presence-scanning has false positives (a rival named in passing does not
make the sentence about the rival) and false negatives (the wrong referent is
often not named nearby at all) relative to the question that actually matters.
This module asks the REFERENT-VERIFICATION question instead: *does this anaphor
denote the target company?* That is the question the human auditor answered to
produce the labels, so it is the only question a judge can be measured on.

WHY A GENERATIVE MODEL AND NOT SOMETHING CHEAPER. The error tail is 68 distinct
referents across 85 errors -- SpaceX, BYD, Rivian, Lightship, Agtonomy, an
inverse-Tesla ETF, a Delaware court filing -- so no curated roster reaches it,
and the project's ticker-agnostic constraint means a roster would have to be
rebuilt for every new target anyway. Ruled out on the same evidence, before any
were built: sentence embeddings (Tesla and SpaceX embed CLOSE, which is exactly
the wrong signal), extractive QA, NLI as the judge, supervised training on 270
labels, and KB entity linking. What is left is a model that already knows, from
pre-training, that Lightship is an RV startup and BYD is a Chinese carmaker.

GATE PER SENTENCE AT THE REWRITE SITE, NOT PER CHAIN. A coref chain can be
internally incoherent -- correct for some mentions and wrong for others -- and
no per-chain verdict can express that. Judging each sentence independently
dissolves the problem: incoherence is never detected, the wrong sentences simply
fail on their own.

FAIL CLOSED, ALWAYS. The judge returns `yes` / `no` / `unsure`, and
coref_eval.accept_only() accepts on `yes` alone. Timeouts, malformed output, a
model that answers in a paragraph, and outright exceptions are all discards.
This is the project's posture applied literally: losing a sentence is
acceptable, corrupting one is not.

CONSTRAINED DECODING RATHER THAN OUTPUT PARSING. The model is given a GBNF
grammar admitting exactly the three answer tokens, so a malformed answer is
impossible by construction instead of being cleaned up afterwards with a
regex over free text. That removes the entire class of "the model said 'Yes,
because...' and the parser took the first word" bugs, and it makes `unsure` a
verdict the model actually chose rather than a bucket the parser fell into.

TEMPERATURE 0 IS NOT REPRODUCIBILITY. It makes one process deterministic; it
does nothing about a llama.cpp bump, a different quantisation, or a rebuilt
wheel. The disk cache in coref_eval.py is what makes a feature set reproducible,
and prompt_version is in its key so editing anything in PROMPT_VERSION's
neighbourhood invalidates the verdicts formed under the old wording.

TICKER-AGNOSTIC. The target is a runtime parameter throughout. The prompt names
the company using COMPANIES[target]["names"][0], the same config entry the rest
of the pipeline resolves aliases from -- there is no per-company prompt text,
no roster of confusable companies, and nothing an engineer must top up when the
pipeline is pointed at a new ticker.

ITS HONEST LIMIT, worth writing down before any result arrives: ticker-agnostic
CODE is not the same claim as ticker-uniform ACCURACY. This judge works by
knowing what Lightship and BYD are. That world knowledge is excellent for
large-cap US equities and decays for obscure targets, and nothing in this module
detects when it has decayed.

API:
    is_available()   -> bool, whether llama-cpp and the model file are usable
    build_prompt()   -> the exact string the model is shown, for one context
    make_judge()     -> a judge callable for coref_eval.evaluate_judge()
"""

from pathlib import Path

from loguru import logger

from stock_predictor.config import COMPANIES, JUDGE_MODEL_PATH, PRIMARY_TICKER

# Bump this whenever ANYTHING below changes the string the model sees -- the
# system prompt, the instruction wording, the answer vocabulary, the context
# rendering. It is part of the verdict cache key in coref_eval.py, so a bump is
# what stops a new prompt being scored with the old prompt's answers. That is
# the single most likely way this stage could produce a fabricated result.
PROMPT_VERSION = "v3"

# INDUSTRY-NEUTRAL WORDING IS PART OF THE TICKER-AGNOSTIC CONSTRAINT, and it is
# easy to violate by accident. v1 of this prompt said "one of {company}'s
# products or vehicles" -- "vehicles" is an automaker assumption that reads as
# nonsense for a chipmaker or a beverage company, and it quietly handed the
# judge a hint that only existed because this corpus's target happens to be a
# carmaker. Ticker-agnostic means the prompt must not encode the target's
# INDUSTRY any more than it encodes the target's competitors. The error classes
# below are therefore named in industry-neutral terms only ("products", "a
# person", "a fund or ETF", "an industry or market", "a generic definition"),
# and a test asserts no sector vocabulary creeps back in.

# The model is constrained to emit exactly one of these three tokens. Kept in
# sync with coref_eval.JUDGE_ANSWERS by the test suite rather than by hope.
ANSWERS = ("yes", "no", "unsure")

# GBNF grammar admitting only the three answers. See the module docstring on why
# this is constrained decoding rather than parsing.
ANSWER_GRAMMAR = 'root ::= "yes" | "no" | "unsure"'

# Context window. MEASURED, not guessed: over the 3,553 coref sentences in the
# corpus the prompt is a median 387 tokens and a p99 of 513, but the maximum is
# 2,446 -- three sentences (0.08%) exceed 2048, and at 2048 llama.cpp raises
# rather than truncating. The 270-row eval set tops out at 513 and so gave no
# warning of this; it under-represents long sentences. 4096 covers every prompt
# in the corpus with margin, at the cost of a larger KV cache.
DEFAULT_N_CTX = 4096

_SYSTEM_PROMPT = (
    "You are a careful financial-news analyst. You decide what a pronoun or "
    "definite description in a news sentence actually refers to, using the "
    "surrounding article as evidence. You answer with one word and nothing else."
)

# The instruction. Three things in it are load-bearing and should not be
# casually reworded:
#
#   * "the marked phrase" -- the judge is asked about ONE specific mention, the
#     one the pipeline is about to overwrite. Asking "is this sentence about
#     the company" is a different, easier and less useful question: a sentence
#     can be about the target while the marked phrase refers to a rival.
#   * the explicit invitation to answer `no` for people, funds, other companies
#     and other companies' products. These are the actual error classes in the
#     labelled set (Musk personally, ETFs holding the target, BYD, SpaceX), and
#     naming the CLASSES rather than the companies is what keeps it
#     ticker-agnostic -- the model supplies the world knowledge that BYD is a
#     different company, the prompt never says so.
#
#     The TARGET'S OWN products go the other way: they answer `yes`. Sentiment
#     about a company's own product is sentiment about the company for a price
#     model -- "the Semi is delayed" is negative information about the target,
#     and discarding it loses real signal. The labelled set was relabelled to
#     match on 2026-08-18 (5 rows across the two eval files); a competitor's
#     product still answers `no`, and so does a target product discussed in
#     another company's context ("Megapack batteries in SpaceX data centers"
#     is a sentence about SpaceX's data centres).
#   * "unsure" being offered at all. A judge with no way to abstain converts
#     every hard case into a confident coin flip, and under accept-on-yes an
#     abstention is a discard -- which is the outcome the project wants for a
#     case the model cannot resolve.
_INSTRUCTION = """Below is a news article extract. One phrase in the marked sentence is wrapped in **double asterisks**.

Question: does the marked phrase refer to {company}, or to something else?

Answer "yes" if the marked phrase refers to {company} itself, or to one of {company}'s own products.
Answer "no" if it refers to anything else -- a different company, a different company's products, a person, a fund or ETF that merely holds {company}, an industry or market, or a generic definition.
Answer "unsure" if the extract does not contain enough evidence to decide.

Answer with exactly one word: yes, no, or unsure."""

# No-span rows carry no marked phrase -- coref tagged the sentence without
# finding a substitutable mention -- so they get a question matched to the
# weaker claim the pipeline is actually making about them. Asking about a
# "marked phrase" that is not there would be incoherent, and silently reusing
# the span wording is how a judge ends up scored on a question it was never
# asked.
_INSTRUCTION_NO_SPAN = """Below is a news article extract. One sentence is marked with >.

Question: is the marked sentence making a statement about {company}?

Answer "yes" if the marked sentence is about {company} itself, or about one of {company}'s own products.
Answer "no" if it is about anything else -- a different company, a different company's products, a person, a fund or ETF that merely holds {company}, an industry or market, a generic definition, or promotional boilerplate.
Answer "unsure" if the extract does not contain enough evidence to decide.

Answer with exactly one word: yes, no, or unsure."""

_MODEL = None
_LOAD_FAILED = False


def company_name(target: str = PRIMARY_TICKER) -> str:
    """The display name for `target`, from the same config the aliases come from.

    Falls back to the ticker itself for a company with no entry -- a judge
    prompted with "TSLA" instead of "Tesla" is worse but still coherent, which
    is the right degradation for a target that has not been described yet.
    """
    entry = COMPANIES.get(target, {})
    names = entry.get("names") or []
    return names[0] if names else target


def build_prompt(ctx, target: str = PRIMARY_TICKER) -> str:
    """The exact string shown to the model for one JudgeContext.

    Separated from inference so the prompt can be inspected, diffed and tested
    without loading 4.7GB of weights -- and so a human can read exactly what the
    judge was asked when a verdict looks wrong.
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

    Memoised for the same reason entity_filter._NLP and coref._MODEL are: the
    load costs seconds and gigabytes, and the corpus pass calls the judge
    thousands of times. _LOAD_FAILED latches so a missing model file warns once
    rather than on every row of a 3-hour run.
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
    """True when llama-cpp imported AND the GGUF file loaded.

    A False here must mean "discard everything", never "raise" -- the same
    enable-gate contract coref.is_available() has.
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

    Returns a function of one JudgeContext returning `yes` / `no` / `unsure`.
    The model is loaded lazily on the first call, so building a judge is free
    and an evaluation that turns out to be fully cached never loads the weights
    at all -- the same property coref.resolve_documents() has.

    Decoding is temperature 0 with the answer grammar attached, so the output is
    one of three tokens by construction. Anything unexpected that still gets
    through -- an empty completion, a backend that ignores the grammar -- lands
    on `unsure`, which is a discard.
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
            # THE JUDGE ITSELF MUST FAIL CLOSED, not just evaluate_judge().
            # This was a real outage: one sentence whose prompt exceeded n_ctx
            # raised ValueError out of llama.cpp and killed a 3.3-hour corpus
            # pass at 67%. The module docstring promised "outright exceptions
            # are all discards" while only the harness delivered that, so any
            # caller driving the judge directly -- which is what a corpus run
            # does -- inherited a crash instead of a discard.
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
