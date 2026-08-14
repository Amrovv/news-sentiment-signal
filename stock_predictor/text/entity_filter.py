"""Sentence-level entity tagging for news articles (FinBERT pre-processing).

This module does NOT filter sentences out of an article (aside from dropping
very short scraper residue, e.g. "Advertisement"). Every remaining sentence is
tagged as being about the target company, about some other named company, or
neither, so that downstream sentiment scoring can be scoped per-entity.

Pipeline:
    split_sentences()  -> sentence-split article bodies with spaCy (batched)
    tag_sentences()     -> per-article: explicit alias tagging + anaphora
    resolve_anaphora()  -> internal helper called by tag_sentences()
    process_articles()  -> batch entry point over a DataFrame of articles
"""

import re

import pandas as pd
import spacy
from loguru import logger
from tqdm import tqdm

from stock_predictor.config import (
    ALIASES,
    ANAPHORA_MAX_GAP,
    GENERIC_ANAPHORA,
    MIN_SENT_CHARS,
    OTHER_COMPANIES,
    SPACY_MODEL,
    SPACY_PIPE_BATCH_SIZE,
)

# Scraped HTML sometimes drops the space after sentence-ending punctuation when
# tags are stripped (e.g. "...record deliveries.Musk said..."), which causes
# spaCy's dependency-parse-based sentencizer to MISS the sentence boundary
# entirely (no whitespace token to hint a break). We insert a space whenever a
# lowercase letter or digit is immediately followed by [.!?] and then an
# uppercase letter, with no space in between. This deliberately does NOT match
# "U.S." ("U.S.Government"): a period preceded by an uppercase letter is left
# alone, since that pattern is far more likely to be an abbreviation than a
# missing-space sentence boundary. It also does not touch "$1.2B" / "Q3 2024.":
# those periods are surrounded by digits, not lowercase-letter-then-uppercase.
_MISSING_SPACE_RE = re.compile(r"(?<=[a-z0-9])([.!?])(?=[A-Z])")


def _fix_missing_space(text: str) -> str:
    """Insert a space after sentence punctuation glued to the next sentence."""
    return _MISSING_SPACE_RE.sub(r"\1 ", text)


def _get_nlp():
    logger.info(f"Loading spaCy model '{SPACY_MODEL}' (excluding ner, lemmatizer)")
    return spacy.load(SPACY_MODEL, exclude=["ner", "lemmatizer"])


def split_sentences(texts: list[str], nlp=None) -> list[list[str]]:
    """Sentence-split a batch of article bodies.

    Loads spacy.load(SPACY_MODEL, exclude=["ner", "lemmatizer"]) once if `nlp`
    is not provided, then runs the whole batch through nlp.pipe() (never a
    per-document loop). Sentences shorter than MIN_SENT_CHARS (after
    stripping whitespace) are dropped as scraper residue.

    Returns one list of sentence strings per input article, in the same
    order as `texts`.
    """
    if nlp is None:
        nlp = _get_nlp()

    cleaned = [_fix_missing_space(t) if isinstance(t, str) else "" for t in texts]

    results: list[list[str]] = []
    for doc in nlp.pipe(cleaned, batch_size=SPACY_PIPE_BATCH_SIZE):
        sents = [s.text.strip() for s in doc.sents]
        sents = [s for s in sents if len(s) >= MIN_SENT_CHARS]
        results.append(sents)
    return results


def _compile_alias_pattern(aliases: list[str]) -> re.Pattern | None:
    """Build a single boundary-anchored, case-insensitive regex from a list
    of alias strings. Each alias is re.escape()'d so multi-word aliases and
    aliases with regex-special characters (e.g. "$TSLA") match literally.

    We use a custom boundary -- (?<![A-Za-z0-9-])...(?![A-Za-z0-9-]) -- rather
    than plain \\b. Plain \\b treats "-" as a non-word character, so a ticker
    embedded in a URL-slug-like fragment (e.g. "tsla-stock-analysis") would
    still count as fully word-boundary-delimited and false-positive. Treating
    "-" as boundary-blocking as well closes that trap while still matching
    normal prose punctuation (spaces, commas, periods).
    """
    if not aliases:
        return None
    escaped = sorted((re.escape(a) for a in aliases), key=len, reverse=True)
    pattern = r"(?<![A-Za-z0-9-])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9-])"
    return re.compile(pattern, re.IGNORECASE)


def _build_ticker_patterns(ticker: str) -> dict[str, re.Pattern | None]:
    alias_cfg = ALIASES[ticker]
    return {
        "unambiguous": _compile_alias_pattern(alias_cfg.get("unambiguous", [])),
        "person": _compile_alias_pattern(alias_cfg.get("person", [])),
        "anaphoric": _compile_alias_pattern(alias_cfg.get("anaphoric", [])),
    }


def _build_other_company_patterns() -> dict[str, re.Pattern]:
    patterns = {}
    for name, aliases in OTHER_COMPANIES.items():
        pat = _compile_alias_pattern(aliases)
        if pat is not None:
            patterns[name] = pat
    return patterns


_GENERIC_ANAPHORA_RE = _compile_alias_pattern(GENERIC_ANAPHORA)


def resolve_anaphora(article_id, sentences: list[str], ticker: str) -> pd.DataFrame:
    """Walk sentences in order, tracking explicit company mentions and
    resolving generic anaphora ("the company", "it", ...) to whichever
    company (target or other) was most recently named explicitly.

    A sentence's target/other tag can come from two sources:
      1. An explicit alias match in that sentence (unambiguous or
         ticker-specific anaphoric phrase for the target; an OTHER_COMPANIES
         alias for an "other" company).
      2. Anaphora resolution: the sentence has no explicit company name of
         its own, but contains a GENERIC_ANAPHORA phrase, so it inherits the
         most-recently-named company. If no company has been named yet, both
         mentions_target and mentions_other stay False and
         resolved_by_anaphora stays False.

    The antecedent decays after ANAPHORA_MAX_GAP sentences with no fresh
    explicit mention: past that gap we stop attributing anaphora to it,
    rather than carrying it forward indefinitely into unrelated later
    content (see module docstring / config.py comment on ANAPHORA_MAX_GAP).

    Returns a DataFrame with columns: article_id, sent_idx, text,
    mentions_target, mentions_other, mentions_ceo, resolved_by_anaphora,
    char_len.
    """
    target_patterns = _build_ticker_patterns(ticker)
    other_patterns = _build_other_company_patterns()

    rows = []
    last_named: str | None = None  # "TARGET", an OTHER_COMPANIES key, or None
    last_named_idx: int | None = None

    for idx, text in enumerate(sentences):
        mentions_target = False
        mentions_other = False
        mentions_ceo = False
        resolved_by_anaphora = False

        explicit_target = bool(
            (target_patterns["unambiguous"] and target_patterns["unambiguous"].search(text))
            or (target_patterns["anaphoric"] and target_patterns["anaphoric"].search(text))
        )
        if target_patterns["person"] and target_patterns["person"].search(text):
            mentions_ceo = True

        explicit_other_name = None
        for name, pat in other_patterns.items():
            if pat.search(text):
                explicit_other_name = name
                break

        if explicit_target:
            mentions_target = True
            last_named = "TARGET"
            last_named_idx = idx
        elif explicit_other_name is not None:
            mentions_other = True
            last_named = explicit_other_name
            last_named_idx = idx
        elif _GENERIC_ANAPHORA_RE and _GENERIC_ANAPHORA_RE.search(text):
            gap = idx - last_named_idx if last_named_idx is not None else None
            if last_named is not None and gap is not None and gap <= ANAPHORA_MAX_GAP:
                if last_named == "TARGET":
                    mentions_target = True
                else:
                    mentions_other = True
                resolved_by_anaphora = True
            # else: no company named yet, or antecedent has decayed -> leave
            # everything False rather than guessing at a stale referent

        rows.append(
            {
                "article_id": article_id,
                "sent_idx": idx,
                "text": text,
                "mentions_target": bool(mentions_target),
                "mentions_other": bool(mentions_other),
                "mentions_ceo": bool(mentions_ceo),
                "resolved_by_anaphora": bool(resolved_by_anaphora),
                "char_len": len(text),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "article_id",
            "sent_idx",
            "text",
            "mentions_target",
            "mentions_other",
            "mentions_ceo",
            "resolved_by_anaphora",
            "char_len",
        ],
    )


def tag_sentences(article_id, sentences: list[str], ticker: str) -> pd.DataFrame:
    """Tag one article's already-split sentences with target/other/ceo/
    anaphora flags. See resolve_anaphora() for the tagging algorithm.

    Design decision (mentions_ceo vs mentions_target): person-tier aliases
    (e.g. "Musk") are ambiguous — a sentence mentioning Musk may be about
    SpaceX or X, not Tesla. mentions_ceo is therefore tracked as an
    independent, informational flag and never sets mentions_target on its
    own. mentions_target is only set by an unambiguous alias match, a
    ticker-specific anaphoric phrase match, or anaphora resolution to a
    previously-named target. This keeps the ambiguous-person signal
    available for downstream analysis without silently inflating the
    target-sentiment set.
    """
    return resolve_anaphora(article_id, sentences, ticker)


def process_articles(df: pd.DataFrame, ticker: str, text_col: str = "processed_body") -> pd.DataFrame:
    """Batch entry point: sentence-split df[text_col] via nlp.pipe, tag each
    article's sentences, and concatenate into one long DataFrame with columns
    article_id, sent_idx, text, mentions_target, mentions_other, mentions_ceo,
    resolved_by_anaphora, char_len.
    """
    logger.info(f"Splitting sentences for {len(df)} articles (ticker={ticker})")
    nlp = _get_nlp()
    texts = df[text_col].tolist()
    article_ids = df["article_id"].tolist()

    all_sentences = split_sentences(texts, nlp=nlp)

    logger.info("Tagging sentences per article")
    frames = []
    for article_id, sentences in tqdm(
        zip(article_ids, all_sentences), total=len(article_ids), desc="Tagging articles"
    ):
        if not sentences:
            continue
        frames.append(tag_sentences(article_id, sentences, ticker))

    if not frames:
        logger.warning("No sentences produced across the whole batch")
        return pd.DataFrame(
            columns=[
                "article_id",
                "sent_idx",
                "text",
                "mentions_target",
                "mentions_other",
                "mentions_ceo",
                "resolved_by_anaphora",
                "char_len",
            ]
        )

    result = pd.concat(frames, ignore_index=True)
    logger.info(f"Produced {len(result)} tagged sentences from {len(df)} articles")
    return result
