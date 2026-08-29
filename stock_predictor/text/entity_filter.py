"""Sentence-level entity tagging for news articles.

Splits article bodies and tags each sentence as being about the target, about
another named company, or neither. Does not filter sentences out, beyond dropping
scraper residue shorter than MIN_SENT_CHARS.

config.COMPANIES is a symmetric registry: any entry can be the target, chosen at
runtime by `ticker`. Its "names" tier is unambiguous; its "person" tier is
informational and never sets mentions_target on its own.

Spans, not strings. The parse that finds sentence boundaries also supplies the
entities tagging needs, so split_sentences() can return spaCy Spans and the tagging
path consumes them. String callers get parsed on demand, keeping one tagging
implementation.

    split_sentences()    sentence-split article bodies with spaCy (batched)
    tag_sentences()      per article: explicit alias tagging + coref mentions
    map_coref_clusters() coref clusters -> per-company mentions
    flag_boilerplate()   corpus-level repeated-sentence flagging
    process_articles()   batch entry point over a DataFrame of articles
"""

import re

from loguru import logger
import pandas as pd
import spacy
from spacy.tokens import Span
from tqdm import tqdm

from stock_predictor.config import (
    BOILERPLATE_MIN_ARTICLES,
    COMPANIES,
    MIN_SENT_CHARS,
    SPACY_EXCLUDE,
    SPACY_MODEL,
    SPACY_MULTIPROCESS_MIN_DOCS,
    SPACY_N_PROCESS,
    SPACY_PIPE_BATCH_SIZE,
    USE_COREF,
)

# Scraped HTML drops the space after sentence-ending punctuation
# ("...deliveries.Musk said..."), which makes spaCy miss the boundary. Skips
# "U.S.Government" and "$1.2B", where the period is not lowercase-then-uppercase.
_MISSING_SPACE_RE = re.compile(r"(?<=[a-z0-9])([.!?])(?=[A-Z])")

# See _get_nlp() on why `ner` must stay enabled.
PERSON_ENT_LABEL = "PERSON"

# Cached spaCy pipeline. Parsing is by far the dominant cost in this module and
# model load is ~1s; the string-input path of tag_sentences() would otherwise
# reload the model on every call.
_NLP = None


def _fix_missing_space(text: str) -> str:
    """Insert a space after sentence punctuation glued to the next sentence."""
    return _MISSING_SPACE_RE.sub(r"\1 ", text)


def _get_nlp():
    """Load (once, then memoise) the spaCy pipeline for splitting and tagging.

    DO NOT ADD "ner" TO THE EXCLUDE LIST. It looks unused and is expensive, but PERSON
    entities are load-bearing in map_coref_clusters(), _coref_hits_in_sentence() and
    _is_person_like(). Emptying doc.ents breaks all three without raising. The same
    goes for "parser", which is what produces doc.sents and the dep_/head fields
    _is_expletive_token() reads. What SPACY_EXCLUDE does drop is the tagger,
    attribute_ruler and lemmatizer, whose pos_/tag_/lemma_ output nothing here reads.
    """
    global _NLP
    if _NLP is None:
        logger.info(f"Loading spaCy model '{SPACY_MODEL}' (excluding {', '.join(SPACY_EXCLUDE)})")
        _NLP = spacy.load(SPACY_MODEL, exclude=SPACY_EXCLUDE)
    return _NLP


def split_sentences(texts: list[str], nlp=None, return_spans: bool = False, n_process=None):
    """Sentence-split a batch of article bodies through nlp.pipe().

    Sentences shorter than MIN_SENT_CHARS after stripping are dropped as scraper
    residue.

    return_spans=False returns lists of strings; True returns spaCy Spans, keeping the
    parse and entities computed to find the boundaries. The same length filter applies
    either way, so both forms select the same sentences.

    `n_process` defaults to SPACY_N_PROCESS on a batch of at least
    SPACY_MULTIPROCESS_MIN_DOCS documents and to 1 below that, since spawning
    workers costs more than it saves on a handful of texts. Workers spawn on
    Windows, so a caller passing more than 1 must be importable and guarded by
    `if __name__ == "__main__"`.

    Returns one list per input article, in input order.
    """
    if nlp is None:
        nlp = _get_nlp()

    cleaned = [_fix_missing_space(t) if isinstance(t, str) else "" for t in texts]

    if n_process is None:
        n_process = SPACY_N_PROCESS if len(cleaned) >= SPACY_MULTIPROCESS_MIN_DOCS else 1
    if n_process > 1:
        logger.info(f"Parsing {len(cleaned)} documents across {n_process} processes")

    results: list[list] = []
    for doc in nlp.pipe(cleaned, batch_size=SPACY_PIPE_BATCH_SIZE, n_process=n_process):
        if return_spans:
            spans = [s for s in doc.sents if len(s.text.strip()) >= MIN_SENT_CHARS]
            results.append(spans)
        else:
            sents = [s.text.strip() for s in doc.sents]
            sents = [s for s in sents if len(s) >= MIN_SENT_CHARS]
            results.append(sents)
    return results


def _compile_alias_pattern(aliases: list[str]) -> re.Pattern | None:
    """Build one boundary-anchored, case-insensitive regex from a list of aliases.

    Each alias is re.escape()'d so multi-word aliases and "$TSLA" match literally.

    The boundary is (?<![A-Za-z0-9-])...(?![A-Za-z0-9-]) rather than \b, because \b
    treats "-" as a non-word character and would match a ticker inside a URL slug
    ("tsla-stock-analysis").
    """
    if not aliases:
        return None
    escaped = sorted((re.escape(a) for a in aliases), key=len, reverse=True)
    pattern = r"(?<![A-Za-z0-9-])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9-])"
    return re.compile(pattern, re.IGNORECASE)


def _build_ticker_patterns(ticker: str) -> dict[str, re.Pattern | None]:
    """Compile the TARGET company's "names" and "person" alias tiers.

    The target is just the COMPANIES entry selected by `ticker` — nothing in
    the registry is target-specific, which is what lets the pipeline run for
    any ticker (TSLA is temporary scaffolding).

    """
    alias_cfg = COMPANIES[ticker]
    return {
        "names": _compile_alias_pattern(alias_cfg.get("names", [])),
        "person": _compile_alias_pattern(alias_cfg.get("person", [])),
    }


def _build_product_keys() -> frozenset[str]:
    """Union of every registry entry's "products" tier, casefolded.

    Not scoped to the target: a product is never the company whoever owns it, and
    scoping would leave a rival's model names treated as company mentions.

    Consumed by is_substitutable_mention().
    """
    keys = set()
    for entry in COMPANIES.values():
        for product in entry.get("products", []) or []:
            key = product.strip().casefold()
            if key:
                keys.add(key)
    return frozenset(keys)


PRODUCT_KEYS = _build_product_keys()


# Job titles and person-denoting head nouns that make an NP name a person rather
# than the company ("the Tesla CEO", "the Tesla billionaire").
_TITLE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"CEO|CFO|CTO|COO|chief executive|co-founder|cofounder|founder|"
    r"chairman|chairwoman|chairperson|president|boss|"
    r"billionaire|billionaires|tycoon|magnate|mogul|entrepreneur"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Copula surface forms accepted by the parse-based expletive test. Matched on
# the token's text because the lemmatizer is excluded from the pipeline (see
# _get_nlp()), so token.lemma_ cannot be relied on here.
_COPULA_FORMS = frozenset({"is", "was", "are", "were", "be", "been", "'s", "’s", "s"})

# Clausal-complement dependency labels: the extraposed clause of an expletive
# "It" construction ("... that both models have ...", "... to note that ...").
_CLAUSAL_DEPS = frozenset({"ccomp", "xcomp", "advcl", "acl", "relcl", "csubj"})


SENTENCE_COLUMNS = [
    "article_id",
    "sent_idx",
    "text",
    "mentions_target",
    "mentions_ceo",
    # Provenance of the tag: coref rather than an explicit name in the sentence.
    "resolved_by_coref",
    "is_boilerplate",
    "char_len",
    # Offsets into this row's `text`, NA when the sentence named the company
    # itself. absa.py replaces exactly these characters.
    "mention_char_start",
    "mention_char_end",
]

# Nullable-integer columns, applied on every frame this module constructs so the
# schema is identical whether or not any sentence was resolved (a plain
# DataFrame of all-None would come back as object/float64).
_SPAN_COLUMN_DTYPES = {"mention_char_start": "Int64", "mention_char_end": "Int64"}


def _empty_sentence_frame() -> pd.DataFrame:
    """An empty frame carrying the exact SENTENCE_COLUMNS schema and dtypes."""
    return pd.DataFrame(columns=SENTENCE_COLUMNS).astype(_SPAN_COLUMN_DTYPES)


def _token_at_char(doc, char_idx: int):
    """Map an absolute character offset in `doc` to the token containing it.

    Regex matches need not align to token boundaries ("$TSLA", hyphenated forms), and
    doc.char_span() returns None on a misaligned range; alignment_mode="expand" covers
    most of it and the idx scan is the fallback.

    Returns None rather than raising: callers read "cannot locate the token" as "not
    an expletive".
    """
    span = doc.char_span(char_idx, char_idx + 1, alignment_mode="expand")
    if span is not None and len(span) > 0:
        return span[0]
    for token in doc:
        if token.idx <= char_idx < token.idx + len(token.text):
            return token
    return None


def _scan_company(pattern: re.Pattern | None, text: str) -> bool:
    """True when a company's alias pattern matches anywhere in one sentence.

    A boundary-anchored regex search, nothing more: the sentence either names the
    company or it does not, and coreference decides everything else.
    """
    if pattern is None:
        return False
    return pattern.search(text) is not None


# Pronouns that can stand in for a COMPANY, never a person. First person plural is
# excluded: in news prose it is almost always inside a quote from a person.
_MENTION_PRONOUNS = frozenset(
    {
        "it",
        "its",
        "they",
        "their",
        "them",
        "theirs",
        "it's",
        "it’s",
        "they're",
        "they’re",
    }
)

# Determiners that may head a short NP ending in a company head noun. A
# determiner alone is handled by _MENTION_PRONOUNS; this set only matters when
# followed by one of _COMPANY_HEAD_NOUNS.
_MENTION_DETERMINERS = frozenset({"the", "this", "its", "their", "our"})

# Head nouns denoting a company or its stock. Deliberately narrow: each entry makes
# "the <word>" substitutable everywhere in the corpus.
_COMPANY_HEAD_NOUNS = frozenset(
    {
        "company",
        "firm",
        "business",
        "corporation",
        "corp",
        "group",
        "conglomerate",
        "powerhouse",
        "giant",
        "maker",
        "automaker",
        "carmaker",
        "chipmaker",
        "manufacturer",
        "automotive",
        "brand",
        "stock",
        "security",
        "outfit",
        "enterprise",
    }
)

# Cap on a determiner-headed referring NP, so "a school bus with its stop arm
# extended" is not read as a company mention on its last word.
_MAX_MENTION_NP_TOKENS = 6

# Matches a trailing possessive clitic ("'s"/"’s") so it can be stripped
# before the bare head noun or product key is compared.
_POSSESSIVE_CLITIC_RE = re.compile(r"(?:'s|’s)$", re.IGNORECASE)


def _strip_possessive(token: str) -> str:
    """Strip a trailing possessive clitic ("'s"/"’s") from one token."""
    return _POSSESSIVE_CLITIC_RE.sub("", token)


def is_substitutable_mention(surface: str) -> bool:
    """True when the company name may be substituted over `surface`.

    Allows a non-person pronoun from _MENTION_PRONOUNS, a short noun phrase headed by
    a company noun from _COMPANY_HEAD_NOUNS (determiner required above one token), or
    a bare company head noun.

    Rejects person pronouns, bare demonstratives, plurals, product names, bare
    possessive clitics, long NPs merely ending in a company word, and generic entity
    nouns such as "the earnings call".

    A rejection means "score the sentence unchanged", never "inject anyway". Consulted
    both here, so a bad span is never chosen, and again in absa.py immediately before
    substituting.
    """
    s = (surface or "").strip()
    if not s:
        return False
    low = s.lower()
    if low in _MENTION_PRONOUNS:
        return True
    tokens = low.split()
    if not tokens or len(tokens) > _MAX_MENTION_NP_TOKENS:
        return False
    head = _strip_possessive(tokens[-1]).strip(".,;:!?()[]\"'“”‘’")
    if head not in _COMPANY_HEAD_NOUNS:
        return False
    if len(tokens) > 1 and _strip_possessive(tokens[0]) not in _MENTION_DETERMINERS:
        return False
    bare = _POSSESSIVE_CLITIC_RE.sub("", low).strip()
    return bare not in PRODUCT_KEYS


def _is_person_like(
    surface: str,
    start: int | None = None,
    end: int | None = None,
    person_spans: list[tuple[int, int]] | None = None,
) -> bool:
    """True when a mention refers to a person rather than to a company.

    Either test suffices: the mention overlaps a PERSON entity from the same document,
    which needs `start`, `end` and `person_spans` in one coordinate system; or its
    surface contains a job-title or person-denoting head noun ("the Tesla CEO"), which
    carries no PERSON entity.

    The surface-only form is what the ABSA path uses, holding the text but not a parse.
    """
    if (
        start is not None
        and end is not None
        and person_spans
        and any(start < p_end and p_start < end for p_start, p_end in person_spans)
    ):
        return True
    return _TITLE_TOKEN_RE.search(surface or "") is not None


def _is_expletive_token(token) -> bool:
    """True when `token` is an expletive ("dummy") subject, from the parse.

    Two cases: dep_ == "expl", which en_core_web_sm assigns only to existential
    "There"; and the extraposition spaCy parses as a plain nsubj, "It" plus a copula
    plus an attr/acomp plus an extraposed clause ("It's no coincidence THAT ...").

    The extraposed clause is what makes the "It" refer to nothing. Without it ("It's a
    good quarter for the company") the pronoun is referential and this is False.
    """
    if token is None:
        return False
    if token.dep_ == "expl":
        return True
    if token.text.lower() != "it" or token.dep_ not in ("nsubj", "nsubjpass"):
        return False
    head = token.head
    if head.text.lower().lstrip("'’") not in _COPULA_FORMS:
        return False
    comps = [c for c in head.children if c.dep_ in ("attr", "acomp")]
    if not comps:
        return False
    clauses = [c for c in head.children if c.dep_ in _CLAUSAL_DEPS]
    for comp in comps:
        clauses.extend(c for c in comp.children if c.dep_ in _CLAUSAL_DEPS)
    for clause in clauses:
        for tok in clause.subtree:
            low = tok.text.lower()
            if low == "that" and tok.dep_ == "mark":
                return True
            if low == "to" and tok.dep_ == "aux":
                return True
    return False


def _is_expletive_mention(doc, abs_start: int, abs_end: int) -> bool:
    """True when the mention at [abs_start, abs_end) in `doc` is an expletive.

    The head token is located by character offset and tested with
    _is_expletive_token().

    Coref links expletive "It" into company chains, and substitution then produces
    "Tesla's no coincidence that ...". Such a mention refers to nothing, so the
    sentence is not resolved at all: no tag, no span, no substitution.
    """
    if doc is None:
        return False
    try:
        if not doc.has_annotation("DEP"):
            return False
    except Exception:  # noqa: BLE001 - a doc-like object without the API
        return False
    span = doc.char_span(abs_start, abs_end, alignment_mode="expand")
    token = None
    if span is not None and len(span) > 0:
        # The mention's head token: its root, or the first "it" it contains
        # (coref mentions occasionally include a trailing clitic, e.g. "It's").
        token = span.root
        for tok in span:
            if tok.text.lower() == "it":
                token = tok
                break
    else:
        token = _token_at_char(doc, abs_start)
    return _is_expletive_token(token)


def _as_spans(sentences, nlp=None) -> list[Span]:
    """Normalize a list of sentences (strings or Spans) to a list of Spans.

    Strings are parsed on demand, which is what gives the module one tagging code
    path: a caller holding only text goes through the same registry matching as the
    batch run.
    """
    if not sentences:
        return []
    if isinstance(sentences[0], Span):
        return list(sentences)
    if nlp is None:
        nlp = _get_nlp()
    texts = [s if isinstance(s, str) else "" for s in sentences]
    return [doc[:] for doc in nlp.pipe(texts, batch_size=SPACY_PIPE_BATCH_SIZE)]


def tag_sentences(
    article_id,
    sentences,
    ticker: str,
    nlp=None,
    coref_mentions=None,
) -> pd.DataFrame:
    """Tag one article's already-split sentences with target and CEO flags.

    `sentences` accepts plain strings or spaCy Spans; strings are parsed on demand, so
    the batch and live paths share one tagging implementation.

    A target tag comes from one of two sources, in strict precedence:

      1. an explicit "names" alias in the sentence, never marked resolved_by_coref
      2. neural coreference via `coref_mentions`, as (abs_char_start, abs_char_end,
         key) triples absolute into the string the Spans were parsed from

    If neither fires the sentence is tagged as about no company, never guessed at.
    mentions_ceo never sets mentions_target on its own.

    Expletive "it" is dropped before it can set a flag or record a span.

    mention_char_start / mention_char_end are relative to this row's `text`, NA when
    the sentence named the company itself or was not resolved.

    `coref_mentions` is meaningful only on the Span path; omitting it tags on explicit
    names alone.

    Returns a SENTENCE_COLUMNS frame. is_boilerplate is always False here.
    """
    spans = _as_spans(sentences, nlp=nlp)
    coref_mentions = list(coref_mentions or [])

    target_patterns = _build_ticker_patterns(ticker)

    # Computed once per Doc: every sentence of an article shares one on the Span
    # path.
    person_spans_by_doc: dict[int, list[tuple[int, int]]] = {}

    def _person_spans(doc) -> list[tuple[int, int]]:
        cached = person_spans_by_doc.get(id(doc))
        if cached is None:
            cached = [
                (ent.start_char, ent.end_char)
                for ent in doc.ents
                if ent.label_ == PERSON_ENT_LABEL
            ]
            person_spans_by_doc[id(doc)] = cached
        return cached

    rows = []
    for idx, span in enumerate(spans):
        raw = span.text
        # Offsets are computed against the STRIPPED text (what we store and
        # regex over), so the leading-whitespace delta must be added back to
        # address the token in the parent doc.
        lead_ws = len(raw) - len(raw.lstrip())
        text = raw.strip()
        doc = span.doc
        offset = span.start_char + lead_ws

        mentions_ceo = False
        resolved_by_coref = False
        mention_span: tuple[int, int] | None = None

        # --- precedence 1: an explicit target name ("names" tier only) ---
        explicit_target = _scan_company(target_patterns["names"], text)

        if target_patterns["person"] and target_patterns["person"].search(text):
            mentions_ceo = True

        mentions_target = explicit_target

        # --- precedence 2: neural coreference, for un-named sentences only ---
        if not explicit_target and coref_mentions:
            keys, coref_span = _coref_hits_in_sentence(
                coref_mentions,
                offset,
                len(text),
                doc=doc,
                text=text,
                person_spans=_person_spans(doc),
            )
            if "TARGET" in keys:
                mentions_target = True
                resolved_by_coref = True
                mention_span = coref_span

        rows.append(
            {
                "article_id": article_id,
                "sent_idx": idx,
                "text": text,
                "mentions_target": bool(mentions_target),
                "mentions_ceo": bool(mentions_ceo),
                "resolved_by_coref": bool(resolved_by_coref),
                "is_boilerplate": False,
                "char_len": len(text),
                "mention_char_start": None if mention_span is None else mention_span[0],
                "mention_char_end": None if mention_span is None else mention_span[1],
            }
        )

    if not rows:
        return _empty_sentence_frame()
    return pd.DataFrame(rows, columns=SENTENCE_COLUMNS).astype(_SPAN_COLUMN_DTYPES)


def _coref_hits_in_sentence(
    coref_mentions: list[tuple[int, int, str]],
    offset: int,
    length: int,
    doc=None,
    text: str = "",
    person_spans: list[tuple[int, int]] | None = None,
) -> tuple[set[str], tuple[int, int] | None]:
    """Company keys whose coref mentions fall inside one sentence, plus the earliest
    substitutable mention's span relative to the sentence text.

    `coref_mentions` carries absolute offsets into the article text; a mention belongs
    to the sentence when fully contained in [offset, offset+length). Containment, not
    overlap: a span straddling a boundary would corrupt the returned offsets.

    Expletive mentions are dropped outright. Person-like and non-substitutable
    mentions still key the sentence but are never chosen as the span, so a sentence
    holding both a junk mention and a good pronoun selects the pronoun.

    `best` is None when every mention is person-like or non-substitutable.
    """
    keys: set[str] = set()
    best: tuple[int, int] | None = None
    for start, end, key in coref_mentions:
        if start < offset or end > offset + length or end <= start:
            continue
        rel = (start - offset, end - offset)
        if _is_expletive_mention(doc, start, end):
            continue
        keys.add(key)
        surface = text[rel[0] : rel[1]]
        if _is_person_like(surface, start, end, person_spans):
            continue
        if not is_substitutable_mention(surface):
            continue
        if best is None or rel[0] < best[0]:
            best = rel
    return keys, best


def flag_boilerplate(
    sentences_df: pd.DataFrame, min_articles: int = BOILERPLATE_MIN_ARTICLES
) -> pd.DataFrame:
    """Set is_boilerplate=True where exact sentence text appears in at least
    `min_articles` distinct articles.

    Catches disclosure notices and syndicated filler that survive the length filter
    and, repeating across hundreds of articles, drag every aggregate toward one value.

    Corpus-level, so the single-article paths leave it False. Returns a copy.
    """
    out = sentences_df.copy()
    if len(out) == 0:
        if "is_boilerplate" not in out.columns:
            out["is_boilerplate"] = pd.Series(dtype=bool)
        return out

    n_articles = out.groupby("text")["article_id"].transform("nunique")
    out["is_boilerplate"] = n_articles >= min_articles
    return out


def map_coref_clusters(
    clusters: list[list[tuple[int, int]]],
    text: str,
    ticker: str,
    patterns=None,
    person_spans: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int, str]], dict[str, int]]:
    """Map one document's coref clusters onto companies.

    A cluster refers to a company when any of its mention surfaces matches that
    company's registry alias pattern. Only the registry is consulted, so this and the
    explicit path cannot disagree about what a name means.

    A cluster matching more than one company is discarded rather than guessed at, and
    counted in stats["ambiguous_clusters"].

    Person mentions never key a cluster. `person_spans` carries PERSON entity offsets
    in the cluster spans' coordinate system, defaulting to None. Title noun phrases
    matching _TITLE_TOKEN_RE ("the Tesla CEO") carry no PERSON entity and are excluded
    on the same grounds. Both rules affect keying only: a person-like mention still
    belongs to its cluster and still tags its sentence when some other mention names
    the company.

    Returns (mentions, stats): a flat list of (abs_char_start, abs_char_end, key) for
    every mention of every unambiguously resolved cluster, and counts of clusters
    seen, resolved, ambiguous and unresolved.
    """
    if patterns is None:
        patterns = _build_ticker_patterns(ticker)["names"]
    target_names = patterns[0] if isinstance(patterns, tuple) else patterns
    person_spans = list(person_spans or [])

    mentions: list[tuple[int, int, str]] = []
    stats = {"clusters": 0, "resolved": 0, "ambiguous": 0, "unresolved": 0}
    for cluster in clusters:
        stats["clusters"] += 1
        keys: set[str] = set()
        for start, end in cluster:
            surface = text[start:end]
            if not surface:
                continue
            if _is_person_like(surface, start, end, person_spans):
                # Unusable for keying; still a member of the cluster, and still
                # tagged below if some other mention resolves the chain.
                continue
            if target_names is not None and target_names.search(surface):
                keys.add("TARGET")
        if not keys:
            stats["unresolved"] += 1
            continue
        stats["resolved"] += 1
        mentions.extend((start, end, "TARGET") for start, end in cluster)
    return mentions, stats


def process_articles(
    df: pd.DataFrame,
    ticker: str,
    text_col: str = "processed_body",
    use_coref: bool = USE_COREF,
) -> pd.DataFrame:
    """Batch entry point: split, tag, concatenate, then flag boilerplate corpus-wide.

    Uses the Span path end to end so the corpus is parsed once.

    With use_coref=True and a working backend, bodies are resolved by
    coref.resolve_documents() and the clusters mapped onto companies; the parse's
    PERSON entities are threaded through to map_coref_clusters(). An unavailable
    backend logs one warning and tags on explicit names alone.

    Coref must see the exact string the Spans were parsed from, or its offsets address
    different characters than the sentence boundaries do. split_sentences() applies
    _fix_missing_space() first, so the cleaned text is what is sent, and the identity
    is asserted against the parsed Doc rather than assumed.

    Columns: SENTENCE_COLUMNS.
    """
    logger.info(f"Splitting sentences for {len(df)} articles (ticker={ticker})")
    nlp = _get_nlp()
    texts = df[text_col].tolist()
    article_ids = df["article_id"].tolist()

    # The single canonical string per article: what coref reads, and (because
    # _fix_missing_space is idempotent) what split_sentences parses.
    cleaned = [_fix_missing_space(t) if isinstance(t, str) else "" for t in texts]

    all_sentences = split_sentences(cleaned, nlp=nlp, return_spans=True)

    coref_mentions_per_article: list[list[tuple[int, int, str]]] = [[] for _ in cleaned]
    if use_coref:
        from stock_predictor.text import coref

        if not coref.is_available():
            logger.warning("Coreference requested but unavailable; tagging on explicit names only")
        else:
            all_clusters = coref.resolve_documents(cleaned)
            patterns = _build_ticker_patterns(ticker)["names"]
            totals = {"clusters": 0, "resolved": 0, "ambiguous": 0, "unresolved": 0}
            n_docs_without_usable = 0
            n_misaligned = 0
            for i, (clusters, sentences) in enumerate(
                tqdm(
                    zip(all_clusters, all_sentences),
                    total=len(all_clusters),
                    desc="Mapping coref clusters",
                )
            ):
                # Explicit alignment check: the Doc the Spans came from must be
                # character-identical to the string coref was given, otherwise
                # every offset below points at the wrong characters.
                if sentences and sentences[0].doc.text != cleaned[i]:
                    n_misaligned += 1
                    continue
                # Same parse as the Spans, so the coordinates match the cluster
                # spans.
                person_spans = (
                    [
                        (ent.start_char, ent.end_char)
                        for ent in sentences[0].doc.ents
                        if ent.label_ == PERSON_ENT_LABEL
                    ]
                    if sentences
                    else []
                )
                mentions, stats = map_coref_clusters(
                    clusters, cleaned[i], ticker, patterns, person_spans=person_spans
                )
                for k, v in stats.items():
                    totals[k] += v
                coref_mentions_per_article[i] = mentions
                if not mentions:
                    n_docs_without_usable += 1
            if n_misaligned:
                logger.warning(
                    f"Coref offsets misaligned for {n_misaligned} articles; "
                    "those are tagged on explicit names only"
                )
            logger.info(
                f"Coref clusters: {totals['clusters']} total, {totals['resolved']} mapped to a "
                f"company, {totals['ambiguous']} discarded as ambiguous (>1 company), "
                f"{totals['unresolved']} naming no registry company; "
                f"{n_docs_without_usable} articles got zero usable clusters"
            )

    logger.info("Tagging sentences per article")
    frames = []
    for article_id, sentences, mentions in tqdm(
        zip(article_ids, all_sentences, coref_mentions_per_article),
        total=len(article_ids),
        desc="Tagging articles",
    ):
        if not sentences:
            continue
        frames.append(
            tag_sentences(
                article_id,
                sentences,
                ticker,
                nlp=nlp,
                coref_mentions=mentions,
            )
        )

    if not frames:
        logger.warning("No sentences produced across the whole batch")
        return _empty_sentence_frame()

    result = pd.concat(frames, ignore_index=True)
    result = flag_boilerplate(result)
    n_boiler = int(result["is_boilerplate"].sum())
    logger.info(
        f"Flagged {n_boiler} boilerplate sentences "
        f"(exact text repeated across >= {BOILERPLATE_MIN_ARTICLES} articles)"
    )
    logger.info(f"Produced {len(result)} tagged sentences from {len(df)} articles")
    return result
