"""Sentence-level entity tagging for news articles (FinBERT pre-processing).

This module does NOT filter sentences out of an article (aside from dropping
very short scraper residue, e.g. "Advertisement"). Every remaining sentence is
tagged as being about the target company, about some other named company, or
neither, so that downstream sentiment scoring can be scoped per-entity.

Config inputs: config.COMPANIES is a single SYMMETRIC registry of companies —
any entry can be the target or an "other" company, chosen at runtime by the
`ticker` argument. The target's entry supplies the "names" (unambiguous) and
"person" (informational only) tiers; every OTHER entry supplies its "names" for
the other-company contrast tag; the "descriptors" tier of EVERY entry feeds
recency-scoped descriptor resolution (see resolve_anaphora()). This is what
makes the pipeline ticker-agnostic: TSLA is temporary scaffolding, and with
ticker="NVDA" Tesla is simply one of the other companies. Also read:
GENERIC_ANAPHORA, ANAPHORA_MAX_GAP, MIN_SENT_CHARS,
BOILERPLATE_MIN_ARTICLES, SPACY_MODEL, SPACY_PIPE_BATCH_SIZE.

Spans, not strings. The dependency parse that splits the sentences is the same
parse that tells us which company a sentence is grammatically ABOUT and which
ORG entities it names, so it is kept rather than thrown away: split_sentences()
can return spaCy Spans (return_spans=True) and the tagging path consumes Spans.
Callers that only have strings (tests, sentiment.analyze()) may still pass
strings — they are parsed on demand so that there is exactly ONE tagging
implementation. See tag_sentences().

Other-company detection is REGISTRY-ONLY. spaCy's NER was briefly used to
supply a generic ORG-based detector for companies the registry does not know;
it was removed once ABSA landed (see _get_nlp() for the full reasoning and for
why the `ner` component itself must stay enabled regardless).

Pipeline:
    split_sentences()  -> sentence-split article bodies with spaCy (batched)
    tag_sentences()     -> per-article: explicit alias tagging + anaphora
    resolve_anaphora()  -> the tagging algorithm itself
    map_coref_clusters() -> coref clusters (coref.py) -> per-company mentions
    flag_boilerplate()  -> corpus-level repeated-sentence flagging
    process_articles()  -> batch entry point over a DataFrame of articles
"""

import re

import pandas as pd
import spacy
from loguru import logger
from spacy.tokens import Span
from tqdm import tqdm

from stock_predictor.config import (
    ANAPHORA_MAX_GAP,
    BOILERPLATE_MIN_ARTICLES,
    COMPANIES,
    GENERIC_ANAPHORA,
    MIN_SENT_CHARS,
    SPACY_MODEL,
    SPACY_PIPE_BATCH_SIZE,
    USE_ANAPHORA_FALLBACK,
    USE_COREF,
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

# Dependency labels that mark the head of a clausal subject in spaCy's English
# scheme. A company "is the subject" when its matched token, or any of that
# token's ancestors, carries one of these — so possessive subjects ("Lucid's
# market debut was led by...") count, since "Lucid" is a poss child of a
# nsubjpass head, while a company inside an oblique/appositive/prepositional
# phrase ("...by Peter Rawlinson, Tesla's former engineer") does not.
SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj"})

# The one spaCy entity label this module consumes. PERSON entities mark
# mentions that must never be treated as the company — see _is_person_like(),
# map_coref_clusters() and _coref_hits_in_sentence(). (ORG was consumed too,
# by the since-removed generic other-company detector; see _get_nlp().)
PERSON_ENT_LABEL = "PERSON"

# Cached spaCy pipeline. Parsing is by far the dominant cost in this module and
# model load is ~1s; the string-input path of tag_sentences() would otherwise
# reload the model on every call.
_NLP = None


def _fix_missing_space(text: str) -> str:
    """Insert a space after sentence punctuation glued to the next sentence."""
    return _MISSING_SPACE_RE.sub(r"\1 ", text)


def _get_nlp():
    """Load (once, then memoise) the spaCy pipeline used for both sentence
    splitting and entity tagging.

    ⚠ DO NOT ADD "ner" TO THE EXCLUDE LIST. Only the lemmatizer is excluded,
    and that is deliberate. NER looks unused now that the ORG-based
    other-company detector is gone, and disabling it is an obvious-looking
    optimisation — it is the second-most expensive component after the parser.
    It is not unused. PERSON entities are load-bearing in three places, and
    silently emptying doc.ents would break all three without raising:

      * map_coref_clusters(person_spans=...) — stops a PERSON's coreference
        chain being keyed to a company through an appositive mention like
        "Tesla CEO Elon Musk". This fixed real corpus bugs (article 140122179,
        where a 36-mention Musk chain was keyed to Tesla).
      * _coref_hits_in_sentence() — stops a person mention being chosen as the
        span the company name is substituted over.
      * _is_person_like(), which both of the above call.

    HISTORY, so this is not re-litigated. NER was excluded originally, turned
    on in Wave 4 to power a generic ORG-based other-company detector
    (_ner_other_companies()), and that detector was later removed: its purpose
    was to stop sentences naming unregistered companies (Avidity Biosciences,
    Samsung) from contaminating the target's sentiment, because FinBERT scores
    a whole sentence and cannot tell whose sentiment it is. ABSA scores toward
    the aspect, which supersedes that justification. The component itself stays
    on for PERSON.
    """
    global _NLP
    if _NLP is None:
        logger.info(f"Loading spaCy model '{SPACY_MODEL}' (excluding lemmatizer; ner enabled)")
        _NLP = spacy.load(SPACY_MODEL, exclude=["lemmatizer"])
    return _NLP


def split_sentences(texts: list[str], nlp=None, return_spans: bool = False):
    """Sentence-split a batch of article bodies.

    Loads the shared pipeline via _get_nlp() if `nlp` is not provided, then
    runs the whole batch through nlp.pipe() (never a per-document loop).
    Sentences shorter than MIN_SENT_CHARS (after stripping whitespace) are
    dropped as scraper residue.

    return_spans=False (default) returns one list of sentence STRINGS per input
    article. return_spans=True returns one list of spaCy Spans instead, keeping
    the dependency parse and the named entities that were computed anyway in
    order to find the sentence boundaries. The same MIN_SENT_CHARS filter is
    applied either way, measured on the stripped span text, so the two forms
    always select the same sentences.

    Returns one list per input article, in the same order as `texts`.
    """
    if nlp is None:
        nlp = _get_nlp()

    cleaned = [_fix_missing_space(t) if isinstance(t, str) else "" for t in texts]

    results: list[list] = []
    for doc in nlp.pipe(cleaned, batch_size=SPACY_PIPE_BATCH_SIZE):
        if return_spans:
            spans = [s for s in doc.sents if len(s.text.strip()) >= MIN_SENT_CHARS]
            results.append(spans)
        else:
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
    """Compile the TARGET company's "names" and "person" alias tiers.

    The target is just the COMPANIES entry selected by `ticker` — nothing in
    the registry is target-specific, which is what lets the pipeline run for
    any ticker (TSLA is temporary scaffolding).

    "descriptors" is deliberately NOT compiled here any more. It is no longer a
    target-explicit tier: descriptors are resolved anaphorically against every
    company that carries them (see _build_descriptor_scopes()).
    """
    alias_cfg = COMPANIES[ticker]
    return {
        "names": _compile_alias_pattern(alias_cfg.get("names", [])),
        "person": _compile_alias_pattern(alias_cfg.get("person", [])),
    }


def _build_descriptor_scopes(ticker: str) -> tuple[re.Pattern | None, dict[str, set[str]]]:
    """Compile every "descriptors" phrase in the registry into one pattern, plus
    a map {lowercased descriptor -> set of antecedent keys that may bear it}.

    The set of companies whose entry lists a phrase IS the resolution scope for
    that phrase: "the automaker" may resolve to Ford, GM, Toyota, VW, Rivian,
    Lucid, NIO, BYD or Tesla, and to nothing else. The target's own key is
    rewritten to the sentinel "TARGET" so the scopes speak the same language as
    the antecedent tracker.
    """
    scopes: dict[str, set[str]] = {}
    for key, entry in COMPANIES.items():
        antecedent_key = "TARGET" if key == ticker else key
        for phrase in entry.get("descriptors", []) or []:
            scopes.setdefault(phrase.strip().lower(), set()).add(antecedent_key)
    pattern = _compile_alias_pattern(sorted(scopes))
    return pattern, scopes


def _build_product_keys() -> frozenset[str]:
    """Union of every registry entry's "products" tier, casefolded.

    Deliberately NOT scoped to the target. A product name is never the company
    itself no matter whose product it is, so Tesla's Megapack and Lucid's Air
    are both suppressed on every run. Scoping this to the target would leave a
    rival's model names ("the first Air sedans") treated as a company anaphor.

    STILL LIVE AFTER THE NER REMOVAL. This tier originally had two consumers:
    suppressing the generic ORG detector (now gone) and
    is_substitutable_anaphor(), which refuses to substitute the company name
    over a product surface. The second consumer is independent of NER and is
    why PRODUCT_KEYS survives — see is_substitutable_anaphor().

    These strings used to be hard-coded in a global NER stoplist. That put one
    company's product catalogue into a shared constant, which contradicts the
    ticker-agnostic constraint the COMPANIES registry exists to satisfy: the
    products belong to a company, so they are configured on that company.
    """
    keys = set()
    for entry in COMPANIES.values():
        for product in entry.get("products", []) or []:
            key = product.strip().casefold()
            if key:
                keys.add(key)
    return frozenset(keys)


PRODUCT_KEYS = _build_product_keys()


_GENERIC_ANAPHORA_RE = _compile_alias_pattern(GENERIC_ANAPHORA)

# Sentence-initial "It"/"Its" only, case-SENSITIVE and anchored at position 0.
# See resolve_anaphora()'s docstring for why bare "it" was demoted from
# GENERIC_ANAPHORA to this much narrower pattern.
_SENT_INITIAL_IT_RE = re.compile(r"^(?:It|Its)\b")

# Job-title tokens that make a noun phrase a TITLE NP naming a PERSON, e.g.
# "the Tesla CEO", "Tesla's co-founder". Used by _is_person_like() as a
# second, surface-level person-like test alongside PERSON entity overlap — see
# map_coref_clusters()'s docstring for why an appositive title NP must never key
# a coreference cluster to the company it names.
#
# The second line is not job TITLES but PERSON-DENOTING HEAD NOUNS of the same
# construction ("the Tesla billionaire", "the EV tycoon"). They were added after
# article 139905684 was adjudicated: a 36-mention Musk chain (every mention
# either "Mr Musk", a pronoun, or a title NP) keyed to TARGET off the single
# mention "the Tesla billionaire's", which the title-token list did not cover.
# The phrase denotes a human being exactly as "the Tesla founder" does, so it is
# the same leak, not a new one.
_TITLE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"CEO|CFO|CTO|COO|chief executive|co-founder|cofounder|founder|"
    r"chairman|chairwoman|chairperson|president|boss|"
    r"billionaire|billionaires|tycoon|magnate|mogul|entrepreneur"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Conservative surface-level expletive ("dummy subject") detector, used ONLY
# when the dependency parse is unavailable — see _is_expletive_mention().
# Matches sentence-initial "It"/"It's" followed by a form of "be" plus a short
# adjective/noun phrase and then a "that"-clause or a "to"-infinitive:
#     "It's no coincidence that both of these models ..."
#     "It is important to note that shares fell."
# and deliberately NOT "It's a good quarter for the company." (no extraposed
# clause), which is a genuinely referential "It".
_EXPLETIVE_IT_FALLBACK_RE = re.compile(
    r"^It(?:'s|’s|\s+(?:is|was|isn't|wasn't|is\s+not|was\s+not|"
    r"has\s+been|had\s+been|would\s+be|will\s+be|seems|appears))\s+"
    r"(?:(?:a|an|the|no|not|any|another|hardly|clearly|also|only|now|still|"
    r"quite|very|too|so|more|less)\s+){0,3}"
    r"[A-Za-z][\w-]*\s+(?:that|to)(?![A-Za-z0-9])"
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
    "target_is_subject",
    "resolved_by_anaphora",
    # Set when the tag came from neural coreference (coref.py) instead of the
    # recency heuristic. The two are mutually exclusive by construction so the
    # mechanisms stay separately measurable — see resolve_anaphora().
    "resolved_by_coref",
    "is_boilerplate",
    "char_len",
    # The mention span that was resolved (either mechanism), as character
    # offsets into this row's `text`, nullable Int64 / NA when the sentence was
    # not resolved anaphorically at all. Forward-looking: aspect-based sentiment
    # needs to substitute the resolved company name into the sentence before
    # target-aware scoring, which needs to know which characters to replace.
    "anaphor_char_start",
    "anaphor_char_end",
]

# Nullable-integer columns, applied on every frame this module constructs so the
# schema is identical whether or not any sentence was resolved (a plain
# DataFrame of all-None would come back as object/float64).
_SPAN_COLUMN_DTYPES = {"anaphor_char_start": "Int64", "anaphor_char_end": "Int64"}


def _empty_sentence_frame() -> pd.DataFrame:
    """An empty frame carrying the exact SENTENCE_COLUMNS schema and dtypes."""
    return pd.DataFrame(columns=SENTENCE_COLUMNS).astype(_SPAN_COLUMN_DTYPES)


def _token_at_char(doc, char_idx: int):
    """Map an absolute character offset in `doc` to the token containing it.

    Regex matches do not necessarily align to token boundaries ("$TSLA" and
    hyphenated forms are the usual culprits), and doc.char_span() returns None
    on a misaligned range. alignment_mode="expand" handles most of that; the
    explicit idx-range scan is the belt-and-braces fallback. Returns None
    rather than raising — callers treat "cannot locate the token" as "not a
    subject", which is the conservative reading.
    """
    span = doc.char_span(char_idx, char_idx + 1, alignment_mode="expand")
    if span is not None and len(span) > 0:
        return span[0]
    for token in doc:
        if token.idx <= char_idx < token.idx + len(token.text):
            return token
    return None


def _token_is_subject(token) -> bool:
    """True when `token` sits inside the sentence's subject subtree."""
    if token is None:
        return False
    if token.dep_ in SUBJECT_DEPS:
        return True
    for ancestor in token.ancestors:
        if ancestor.dep_ in SUBJECT_DEPS:
            return True
    return False


def _match_is_subject(doc, char_idx: int) -> bool:
    """Subject test for a regex match, addressed by absolute char offset."""
    try:
        return _token_is_subject(_token_at_char(doc, char_idx))
    except (ValueError, IndexError):  # never let parse/offset weirdness escape
        return False


def _scan_company(pattern: re.Pattern | None, text: str, doc, offset: int):
    """Find a company's mentions in one sentence.

    Returns (earliest_match_start_in_text, matched_in_subject_position), or
    (None, False) when the company is not mentioned. A company counts as a
    subject if ANY of its matches is in subject position, while the position
    used for the fallback precedence rule is always the earliest match.
    """
    if pattern is None:
        return None, False
    earliest = None
    is_subject = False
    for m in pattern.finditer(text):
        if earliest is None:
            earliest = m.start()
        if not is_subject and _match_is_subject(doc, offset + m.start()):
            is_subject = True
    return earliest, is_subject


# Closed-class pronouns that can genuinely stand in for a COMPANY (never a
# person): third-person-plural/neuter "it"/"they" and their inflections.
# "he"/"she"/"i"/"you"/"his"/"her" are DELIBERATELY absent -- see
# is_substitutable_anaphor()'s docstring.
#
# First-person-plural ("we"/"our"/"us"/"ours"/"we're") used to be here too, on
# the theory that it could be the company speaking of itself. A context-aware
# audit of the corpus substitutions found 10 cases where that produced
# subject-verb disagreement -- "'We want the future to look like the future,'
# Musk said" became "Tesla want the future to look like the future" -- because
# in news prose a first-person plural is almost always inside a quote from a
# PERSON, not the company speaking in its own voice. Removed accordingly.
_ANAPHORIC_PRONOUNS = frozenset({
    "it", "its", "they", "their", "them", "theirs",
    "it's", "it’s", "they're", "they’re",
})

# Determiners that may head a short noun phrase ending in a company-denoting
# head noun ("the company", "this firm", "its business"). A demonstrative or
# possessive determiner ALONE ("this", "that", "its") is handled by
# _ANAPHORIC_PRONOUNS above; this set only matters when the determiner is
# followed by one of _COMPANY_HEAD_NOUNS.
_ANAPHORIC_DETERMINERS = frozenset({"the", "this", "its", "their", "our"})

# Head nouns that denote a company (or its stock) rather than any other kind of
# entity a coref error might have dragged into a chain. Deliberately narrow:
# adding a word here means "the <word>" and "<word>" alone become substitutable
# anywhere in the corpus, so each entry earns its place by being a genuine,
# common way news prose refers back to "the company" without naming it.
_COMPANY_HEAD_NOUNS = frozenset({
    "company", "firm", "business", "corporation", "corp", "group",
    "conglomerate", "powerhouse", "giant", "maker", "automaker", "carmaker",
    "chipmaker", "manufacturer", "automotive", "brand", "stock", "security",
    "outfit", "enterprise",
})

# A determiner-headed anaphoric NP is capped at this many tokens. Short spans
# ("the EV maker", "the electric vehicle (ev) maker") are genuinely anaphoric;
# a long NP that happens to end in a company word ("a school bus with its stop
# arm extended and red lights flashing") is not -- its LAST WORD is not what
# the phrase is about, "its" is, and that "its" is a possessive modifying
# "stop arm", not a company reference. Capping length is a cheap, effective
# guard against exactly that shape of coref/parse error.
_MAX_ANAPHOR_NP_TOKENS = 6

# Matches a trailing possessive clitic ("'s"/"’s") so it can be stripped
# before the bare head noun or product key is compared.
_POSSESSIVE_CLITIC_RE = re.compile(r"(?:'s|’s)$", re.IGNORECASE)


def _strip_possessive(token: str) -> str:
    """Strip a trailing possessive clitic ("'s"/"’s") from one token."""
    return _POSSESSIVE_CLITIC_RE.sub("", token)


def is_substitutable_anaphor(surface: str) -> bool:
    """True when `surface` is a referring expression the company's name may be
    SUBSTITUTED over, in either entity_filter's span selection or absa.py's
    injection step (absa.py imports this directly).

    WHAT IT ALLOWS:
      * a closed-class, non-person anaphoric pronoun in _ANAPHORIC_PRONOUNS
        ("it", "its", "they", "we", "our", and the "it's"/"they're"/"we're"
        contractions);
      * a short noun phrase (at most _MAX_ANAPHOR_NP_TOKENS tokens) headed by a
        company-denoting noun in _COMPANY_HEAD_NOUNS, optionally preceded by an
        anaphoric determiner ("the company", "this firm", "its business",
        "the automaker's") -- the determiner is REQUIRED whenever the phrase
        has more than one token, so "big company" or "publicly traded firm"
        (an anaphoric-looking NP with a non-anaphoric modifier) is rejected;
      * a bare company head noun with no determiner at all ("Company", as
        headline/caption style sometimes renders it).

    WHAT IT REJECTS, AND WHY EACH ONE MATTERS. This function exists because 442
    of 2,845 corpus substitutions injected the company name over something that
    was not a company anaphor at all -- real breakage, not a hypothetical:

      * PERSON pronouns ("he", "she", "I", "you", "his", "her") are simply
        absent from _ANAPHORIC_PRONOUNS -- the omission IS the rejection, no
        separate check needed. "He he led the finance team there." must never
        become "Tesla he led the finance team there.".
      * a demonstrative alone ("this", "that") is not a noun phrase and denotes
        nothing on its own -- "Let's get what we can" must never become
        "Let{NAME}'s get what we can" because "'s" tail-matched some pronoun
        rule.
      * PLURAL noun phrases ("both companies", "the companies", "these
        companies") denote MORE than one firm, so substituting a single
        company's name is simply wrong regardless of which one: "Both stocks
        trade at steep valuations" must not become "Tesla trade at steep
        valuations".
      * a proper noun the coreference model mistakenly dragged into the
        target's chain ("Brazil", in "That list is Australia, Brazil, India,
        Korea, and Vietnam.") is not covered by any allowed pattern here, since
        it is neither a closed-class pronoun nor headed by a company noun.
      * a PRODUCT name from the registry's "products" tier (PRODUCT_KEYS) is
        rejected even when its surface would otherwise parse as an anaphoric
        NP, because a product is not the company itself.
      * a possessive clitic on its own ("'s") has no head noun to test at all
        and falls out of the token/head checks.
      * an arbitrary long NP that happens to end in a company word ("a school
        bus with its stop arm extended and red lights flashing") is rejected
        by the _MAX_ANAPHOR_NP_TOKENS cap: "...whether FSD would stop for a
        school bus with its stop arm extended..." must not become "...would
        stop for Tesla.".
      * a generic entity noun with no company sense ("the earnings call", "the
        vehicle", "the shares") is rejected because its head is not in
        _COMPANY_HEAD_NOUNS: "It was a remarkable moment in the earnings
        call." must not become "...a remarkable moment in Tesla.".

    A REJECTION HERE MEANS "SCORE THE SENTENCE WITH ITS TEXT UNCHANGED", NEVER
    "INJECT ANYWAY". This function is consulted at TWO points precisely so that
    a bad span can never sneak an injection past it: entity_filter uses it to
    keep a non-substitutable mention from ever being CHOSEN as the anaphor
    span (see _coref_hits_in_sentence()), and absa.py uses it again,
    defensively, immediately before the actual text substitution (see
    _substitute_resolved()). Either check failing puts the sentence in exactly
    the position FinBERT is already in -- scored without an explicit aspect
    anchor -- which is always preferable to scoring a sentence that now reads
    as nonsense.
    """
    s = (surface or "").strip()
    if not s:
        return False
    low = s.lower()
    if low in _ANAPHORIC_PRONOUNS:
        return True
    tokens = low.split()
    if not tokens or len(tokens) > _MAX_ANAPHOR_NP_TOKENS:
        return False
    head = _strip_possessive(tokens[-1]).strip(".,;:!?()[]\"'“”‘’")
    if head not in _COMPANY_HEAD_NOUNS:
        return False
    if len(tokens) > 1 and _strip_possessive(tokens[0]) not in _ANAPHORIC_DETERMINERS:
        return False
    bare = _POSSESSIVE_CLITIC_RE.sub("", low).strip()
    if bare in PRODUCT_KEYS:
        return False
    return True


def _is_person_like(
    surface: str,
    start: int | None = None,
    end: int | None = None,
    person_spans: list[tuple[int, int]] | None = None,
) -> bool:
    """True when a mention refers to a PERSON rather than to a company.

    Two independent tests, either of which is sufficient:
      * the mention's char range overlaps a PERSON entity spaCy found in the
        same document ("Tesla CEO Elon Musk", "Mr Musk") — only available when
        `start`, `end` and `person_spans` are supplied, in the SAME coordinate
        system;
      * the mention's surface contains a job-title or person-denoting head noun
        ("the Tesla CEO", "the Tesla billionaire's"), which is a title NP naming
        a person via their company and carries no PERSON entity for the first
        test to catch.

    Two callers, for two different reasons, both of which reduce to "a person
    mention must never be treated as the company":
      * map_coref_clusters() — a person mention may not KEY a coref cluster to a
        company (see that docstring);
      * absa.build_pairs() / _coref_hits_in_sentence() — a person mention may
        never be the span the company name is SUBSTITUTED over. "It also claims
        Mr Musk had himself suggested ..." must not become "It also claims Tesla
        had himself suggested ...".
    The surface-only form (no spans) is what the ABSA path uses, since it holds
    the sentence text but not the parse.
    """
    if start is not None and end is not None and person_spans:
        if any(start < p_end and p_start < end for p_start, p_end in person_spans):
            return True
    return _TITLE_TOKEN_RE.search(surface or "") is not None


def _is_expletive_token(token) -> bool:
    """True when `token` is an expletive ("dummy") subject, from the parse.

    Two cases:
      * dep_ == "expl" — spaCy's own expletive label. In practice en_core_web_sm
        only assigns it to existential "There", never to "It".
      * the extraposition construction spaCy DOES parse as a plain nsubj:
        "It" + a copula + an attr/acomp complement + an extraposed clause
        ("It's no coincidence THAT both of these models ...", "It is important
        TO note ..."). The extraposed clause is what makes the "It" refer to
        nothing; without it ("It's a good quarter for the company") the pronoun
        is genuinely referential and this returns False.
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


def _is_expletive_mention(
    doc, abs_start: int, abs_end: int, sentence_text: str, rel_start: int = 0
) -> bool:
    """True when the mention at [abs_start, abs_end) in `doc` is an expletive.

    THE PARSE IS PREFERRED whenever it is available: the mention's head token is
    located by character offset and tested with _is_expletive_token(), which
    reads dep_ labels and the extraposition structure. Only when there is no
    usable parse (a `doc` without dependency annotation — the string-input path
    when a caller supplies a parser-less pipeline) does this fall back to the
    conservative surface regex _EXPLETIVE_IT_FALLBACK_RE, which requires a
    sentence-initial "It"/"It's", a form of "be", a short adjective/noun phrase
    and an extraposed "that"/"to" clause. The fallback is only consulted for a
    mention that STARTS the sentence (`rel_start` == 0), since the pattern is
    anchored there and says nothing about a mid-sentence mention.

    WHY THIS EXISTS. Coreference happily links an expletive "It" into a company
    chain, and the ABSA substitution then injects the company name over it:
    "It's no coincidence that both of these models have starting prices under
    $50,000." became "Tesla's no coincidence that ...". The mention refers to
    nothing, so the sentence is not anaphora-resolved at all — no tag, no span,
    no substitution.
    """
    if doc is not None:
        try:
            has_parse = doc.has_annotation("DEP")
        except Exception:  # noqa: BLE001 - a doc-like object without the API
            has_parse = False
        if has_parse:
            span = doc.char_span(abs_start, abs_end, alignment_mode="expand")
            token = None
            if span is not None and len(span) > 0:
                # The mention's head token: its root, or the first "it" it
                # contains (coref mentions occasionally include a trailing
                # clitic, e.g. "It's").
                token = span.root
                for tok in span:
                    if tok.text.lower() == "it":
                        token = tok
                        break
            else:
                token = _token_at_char(doc, abs_start)
            return _is_expletive_token(token)
    if rel_start != 0:
        return False
    return _EXPLETIVE_IT_FALLBACK_RE.match(sentence_text or "") is not None


def _as_spans(sentences, nlp=None) -> list[Span]:
    """Normalize a list of sentences (strings OR Spans) to a list of Spans.

    This is what gives the module exactly ONE tagging code path. Strings are
    parsed on demand (one nlp.pipe() batch, whole-document span per sentence)
    so that a caller holding only text — tests, and sentiment.analyze() on the
    live-demo path — goes through precisely the same dependency-based subject
    detection and registry matching as the batch corpus run. Any
    string-only shortcut here would be train/serve skew by construction: the
    demo would tag a sentence differently from the pipeline the model was
    trained on.
    """
    if not sentences:
        return []
    if isinstance(sentences[0], Span):
        return list(sentences)
    if nlp is None:
        nlp = _get_nlp()
    texts = [s if isinstance(s, str) else "" for s in sentences]
    return [doc[:] for doc in nlp.pipe(texts, batch_size=SPACY_PIPE_BATCH_SIZE)]


def resolve_anaphora(
    article_id,
    sentences,
    ticker: str,
    nlp=None,
    coref_mentions=None,
    use_anaphora_fallback: bool = USE_ANAPHORA_FALLBACK,
) -> pd.DataFrame:
    """Walk sentences in order, tracking explicit company mentions and
    resolving anaphora (generic "the company"/sentence-initial "It", and
    scoped descriptors like "the automaker") to whichever eligible company was
    most recently named explicitly.

    `sentences` may be a list of strings or a list of spaCy Spans; strings are
    parsed on demand (see _as_spans()).

    A sentence's target/other tag can come from two sources:
      1. An explicit mention in that sentence: a "names" alias of the target,
         or a "names" alias of any other COMPANIES entry. Other-company
         detection is REGISTRY-ONLY — see the note below.
      2. Anaphora resolution: the sentence has no explicit company name of
         its own, but contains a generic anaphor or a company descriptor, so
         it inherits a previously-named company. If nothing eligible has been
         named, both mentions_target and mentions_other stay False and
         resolved_by_anaphora stays False.

    Independent flags (was: an if/elif chain). mentions_target and
    mentions_other are computed independently, so a sentence naming both the
    target and another company gets BOTH flags. The old chain set only
    mentions_target for such sentences and always handed the anaphora
    antecedent to the target, which produced a documented failure mode: in a
    Lucid-focused article ("Lucid's market debut ... was led by Peter
    Rawlinson, Tesla's former chief vehicle engineer"), an incidental
    possessive Tesla mention flipped the antecedent to TARGET and mistagged
    the five following Lucid sentences as Tesla sentences (article 139579101,
    notebook 2.0 section 1.4).

    is_comparative marks the both-companies case. Sentiment attribution is
    unreliable there — FinBERT scores the sentence as a whole, so "X
    outperformed Tesla" reads positive even though the positive sentiment
    belongs to X. Downstream aggregation keeps a comparative-excluded mean
    alongside the plain one rather than silently picking a side here.

    OTHER-COMPANY DETECTION IS REGISTRY-ONLY. A generic detector over spaCy's
    ORG entities used to run whenever the registry found nothing, tagging
    mentions_other for companies the registry does not know (Avidity
    Biosciences, Samsung) and setting other_source="ner". It was removed.

    Its justification was that FinBERT scores a whole SENTENCE and cannot tell
    whose sentiment it is, so a sentence about an unregistered rival polluted
    the target's mean and had to be detected in order to be contrasted away.
    ABSA scores sentiment TOWARD AN ASPECT, which addresses that directly and
    supersedes the justification. What the detector still did in practice was
    populate mentions_other for the sent_other_mean_* / absa_other_mean_*
    feature family, at a measured precision of ~50-65% — noise documented in
    notebook 2.3 and never validated. Of 10,502 NER-sourced non-boilerplate
    rows, only 1,369 also carried mentions_target (the contamination case ABSA
    now handles); the other 9,133 existed purely to feed that family.

    Consequences, so they are not mistaken for regressions: mentions_other
    falls substantially, other_source is now only "registry" or "",
    needs_score() shrinks so runs get cheaper, and sent_other_mean_* becomes
    precise but narrower, with more articles NaN.

    is_comparative was ALREADY registry-only before the removal and is
    unchanged by it. The reasoning is worth keeping because it generalises:
    mentions_other is ADDITIVE — a false positive merely adds a sentence to a
    contrast set — whereas is_comparative drives an EXCLUSION
    (sent_entity_excl_comp_*), so every false positive DELETES a genuine, often
    strongly polarised, target sentence from the feature. The two costs are not
    symmetric, so the two gates never shared an evidence bar.

    Antecedent update rules. Only a sentence with at least one EXPLICIT
    company mention may update the antecedent, and then:
      * If the sentence ends in "?" it does NOT update it. Rhetorical/promo
        questions ("Missed Nvidia and Tesla?") name companies without
        establishing a topic; letting them set the antecedent hijacked the
        following paragraphs. The sentence's own flags are still set.
      * If exactly one company is mentioned, it becomes the antecedent.
      * If several are mentioned, the one matching in SUBJECT position wins.
        This replaces the old first-mention-wins rule, which was only ever a
        positional proxy for "which company is this sentence about" adopted
        because no parse was available. Now that the parse is kept, the real
        signal is used: "According to Ford, Tesla is expanding its Berlin
        plant" is a sentence about Tesla even though Ford is named first.
      * If several companies are subjects, or none is, we fall back to the old
        earliest-match-position rule, which still handles the Lucid case above
        (incidental mentions sit late, in trailing subordinate clauses).

    (A NER-sourced other company used to be admitted as an antecedent only in
    subject position, on the grounds that a merely-named ORG inside an
    attribution — "...according to data tracked by Wells Fargo's Colin Langan"
    — is not what the sentence is about. That rule went with the detector. The
    registry is precise enough that a mention is evidence of topic, so registry
    candidates anchor unconditionally and subject-wins sorts out which one.)

    Generic anaphora fires only when the sentence has NO explicit company
    mention, and requires either a GENERIC_ANAPHORA phrase anywhere in the
    sentence or a sentence-initial "It"/"Its". Bare "it" used to be a
    GENERIC_ANAPHORA entry matched anywhere in the sentence; that was far too
    promiscuous ("it is worth noting", "analysts said it was hard to
    believe") and was a major precision hole, so it was demoted to the
    case-sensitive sentence-initial form, which is the position where "It" is
    genuinely anaphoric to the previous sentence's subject.

    Descriptor resolution ("the automaker", "the chipmaker") is a SCOPED
    variant of the same mechanism and takes priority over the generic anaphor
    in a sentence carrying both. A descriptor resolves to the most recently
    named company whose OWN registry entry lists that descriptor, subject to
    the same ANAPHORA_MAX_GAP decay; if the nearest qualifying company is out
    of range, or none has been named, the sentence resolves to NEITHER. It
    never falls back to the target. Descriptors used to be hard explicit
    target matches, which is wrong in both directions: in a Ford-heavy article
    "the automaker" was scored as Tesla, and under ticker="F" the phrase kept
    its Tesla reading. Expect mentions_target to fall as a result — that is
    the correction, not a regression. Because the resolution is anaphoric, it
    sets resolved_by_anaphora=True and leaves other_source empty (nothing was
    explicitly named).

    The antecedent decays after ANAPHORA_MAX_GAP sentences with no fresh
    explicit mention: past that gap we stop attributing anaphora to it,
    rather than carrying it forward indefinitely into unrelated later
    content (see module docstring / config.py comment on ANAPHORA_MAX_GAP).

    target_is_subject records whether the TARGET matched in subject position
    in this sentence (always False when the target was only reached by
    anaphora). It is informational: a sentence whose grammatical subject is
    the target is far more likely to be genuinely about the target, which is
    useful both for downstream weighting and for the aspect-based sentiment
    work that follows.

    COREFERENCE (optional, `coref_mentions`). When neural coreference is
    enabled, process_articles() resolves the whole article body first and passes
    the mentions it could attribute to a company as a list of
    (abs_char_start, abs_char_end, key) triples, where `key` is "TARGET" or a
    COMPANIES key and the offsets are absolute into the SAME string the Spans
    were parsed from. Those mentions fill exactly the slot the heuristic fills,
    under a strict precedence:

      1. An EXPLICIT alias/NER match in the sentence itself always wins and is
         never marked resolved_by_coref. Coref only speaks for sentences that
         name no company of their own.
      2. Otherwise, if any coref mention from a company-resolved cluster falls
         inside the sentence, its key sets mentions_target/mentions_other and
         resolved_by_coref=True.
      3. Otherwise the heuristic runs unchanged (scoped descriptors, then
         generic anaphora / sentence-initial "It", with the decay window), and
         sets resolved_by_anaphora as it always has.

    resolved_by_coref and resolved_by_anaphora can therefore never both be True
    for one sentence, which is what keeps the two mechanisms independently
    measurable rather than merged into one unattributable number.

    USE_ANAPHORA_FALLBACK (`use_anaphora_fallback`, default config.USE_ANAPHORA_
    FALLBACK = False). Gates the ENTIRE heuristic block below -- both scoped
    descriptor resolution and generic anaphora/sentence-initial "It" -- and
    only that block; explicit matches (precedence 1) and coref (precedence 2)
    are unaffected. A context-aware audit measured the heuristic's precision at
    13% against coref's 74% on the same task, so with the flag at its default
    a sentence that would only have been resolved by the heuristic simply gets
    no tag: resolved_by_anaphora stays False and no span is recorded, exactly
    as if the sentence named no company at all. See config.py for the full
    evidence and the reasoning for keeping the capability behind a flag rather
    than deleting it.

    EXPLETIVE "IT" IS NOT AN ANAPHOR. Both resolution mechanisms above can be
    handed a dummy subject — coref links it into a company chain, and the
    sentence-initial "It" rule matches it by position — and both are wrong the
    same way: "It's no coincidence that both of these models have starting
    prices under $50,000." is about nothing, and injecting the company name over
    the pronoun downstream produced "Tesla's no coincidence that ...". A mention
    _is_expletive_mention() recognises (parse-preferred, regex fallback) is
    therefore dropped before it can set any flag or record any span.

    anaphor_char_start / anaphor_char_end record the resolved mention's span
    RELATIVE TO THIS ROW'S `text`, from whichever mechanism fired (NA when
    neither did). Downstream aspect-based scoring substitutes the resolved
    company name into the sentence, and needs to know which characters to
    replace.

    Returns a DataFrame with the SENTENCE_COLUMNS schema. is_boilerplate is
    always False here — it is corpus-level and set by flag_boilerplate().
    """
    spans = _as_spans(sentences, nlp=nlp)
    coref_mentions = list(coref_mentions or [])

    target_patterns = _build_ticker_patterns(ticker)
    descriptor_re, descriptor_scopes = _build_descriptor_scopes(ticker)

    # PERSON entities per parsed Doc, computed once per document rather than per
    # sentence (every sentence of an article shares one Doc on the Span path).
    # They mark mentions that must never be substituted over — see
    # _coref_hits_in_sentence().
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
    last_named: str | None = None  # "TARGET", a non-target key, or None
    last_named_idx: int | None = None
    # Per-company recency, needed by scoped descriptor resolution: "the
    # automaker" must look past a more recent Nvidia mention to the last
    # company that actually bears that descriptor.
    last_named_by_key: dict[str, int] = {}

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
        resolved_by_anaphora = False
        resolved_by_coref = False
        anaphor_span: tuple[int, int] | None = None

        # --- explicit target ("names" tier only) ---
        target_pos, target_is_subject = _scan_company(
            target_patterns["names"], text, doc, offset
        )
        explicit_target = target_pos is not None

        if target_patterns["person"] and target_patterns["person"].search(text):
            mentions_ceo = True

        mentions_target = explicit_target

        if explicit_target:
            if not text.endswith("?"):
                # Rhetorical/promo question -> the flag stands, but the
                # antecedent is untouched (and no recency recorded).
                last_named, last_named_idx = "TARGET", idx
                last_named_by_key["TARGET"] = idx
            continue_to_anaphora = False
        else:
            continue_to_anaphora = True

        # --- neural coreference (precedence 2: only for un-named sentences) ---
        if continue_to_anaphora and coref_mentions:
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
                anaphor_span = coref_span
                continue_to_anaphora = False

        if continue_to_anaphora and use_anaphora_fallback:
            descriptor_key, descriptor_span = _resolve_descriptor(
                text, idx, descriptor_re, descriptor_scopes, last_named_by_key
            )
            if descriptor_key == "TARGET":
                mentions_target = True
                resolved_by_anaphora = True
                anaphor_span = descriptor_span
            elif descriptor_re is None or descriptor_re.search(text) is None:
                # No descriptor in this sentence at all -> the generic anaphor
                # path applies. A sentence that DID carry a descriptor but
                # failed to resolve deliberately stops here rather than falling
                # through to the generic rule, which would hand it to whatever
                # was last named -- usually the target, i.e. exactly the bug
                # scoped descriptors exist to fix.
                generic_m = _GENERIC_ANAPHORA_RE.search(text) if _GENERIC_ANAPHORA_RE else None
                initial_it_m = _SENT_INITIAL_IT_RE.match(text)
                if (
                    initial_it_m is not None
                    and generic_m is None
                    and _is_expletive_mention(
                        doc,
                        offset + initial_it_m.start(),
                        offset + initial_it_m.end(),
                        text,
                        initial_it_m.start(),
                    )
                ):
                    # Expletive "It" — a dummy subject referring to nothing. It
                    # is not an anaphor, so the sentence resolves to no company
                    # at all (and nothing is substituted downstream).
                    initial_it_m = None
                if generic_m or initial_it_m:
                    gap = idx - last_named_idx if last_named_idx is not None else None
                    if last_named is not None and gap is not None and gap <= ANAPHORA_MAX_GAP:
                        if last_named == "TARGET":
                            mentions_target = True
                            resolved_by_anaphora = True
                        # Prefer the generic phrase ("the company") over a bare
                        # sentence-initial "It" when both are present: it is the
                        # more informative span to substitute into later.
                        anaphor_m = generic_m or initial_it_m
                        anaphor_span = (anaphor_m.start(), anaphor_m.end())
                    # else: no company named yet, or antecedent has decayed ->
                    # leave everything False rather than guessing at a stale
                    # referent

        rows.append(
            {
                "article_id": article_id,
                "sent_idx": idx,
                "text": text,
                "mentions_target": bool(mentions_target),
                "mentions_ceo": bool(mentions_ceo),
                "target_is_subject": bool(target_is_subject),
                "resolved_by_anaphora": bool(resolved_by_anaphora),
                "resolved_by_coref": bool(resolved_by_coref),
                "is_boilerplate": False,
                "char_len": len(text),
                "anaphor_char_start": None if anaphor_span is None else anaphor_span[0],
                "anaphor_char_end": None if anaphor_span is None else anaphor_span[1],
            }
        )

    if not rows:
        return _empty_sentence_frame()
    return pd.DataFrame(rows, columns=SENTENCE_COLUMNS).astype(_SPAN_COLUMN_DTYPES)


def _resolve_descriptor(
    text: str,
    idx: int,
    descriptor_re: re.Pattern | None,
    descriptor_scopes: dict[str, set[str]],
    last_named_by_key: dict[str, int],
) -> tuple[str | None, tuple[int, int] | None]:
    """Resolve a descriptor phrase in `text` to an antecedent key, or None.

    Considers every descriptor the sentence contains and picks the most
    recently named company that BEARS one of them (ties broken by recency
    across descriptors), subject to the ANAPHORA_MAX_GAP decay. Returns
    (None, None) when the sentence has no descriptor, when no company in the
    descriptor's scope has been named, or when the nearest such company is out
    of range — the caller must then tag neither company, never the target by
    default.

    Also returns the (start, end) offsets of the WINNING descriptor phrase
    within `text`, which the caller records as anaphor_char_start/end for the
    downstream aspect-based substitution.
    """
    if descriptor_re is None:
        return None, None
    best_key: str | None = None
    best_idx: int | None = None
    best_span: tuple[int, int] | None = None
    for m in descriptor_re.finditer(text):
        for key in descriptor_scopes.get(m.group(0).lower(), ()):
            named_at = last_named_by_key.get(key)
            if named_at is None:
                continue
            if best_idx is None or named_at > best_idx:
                best_key, best_idx = key, named_at
                best_span = (m.start(), m.end())
    if best_key is None or best_idx is None:
        return None, None
    if idx - best_idx > ANAPHORA_MAX_GAP:
        return None, None
    return best_key, best_span


def _coref_hits_in_sentence(
    coref_mentions: list[tuple[int, int, str]],
    offset: int,
    length: int,
    doc=None,
    text: str = "",
    person_spans: list[tuple[int, int]] | None = None,
) -> tuple[set[str], tuple[int, int] | None]:
    """Company keys whose coref mentions fall inside one sentence, plus the
    earliest SUBSTITUTABLE mention's span RELATIVE to the sentence text.

    `coref_mentions` carries ABSOLUTE character offsets into the article text
    the sentence Spans were parsed from; `offset` is the absolute start of the
    stripped sentence and `length` its character length, so a mention belongs to
    the sentence exactly when it is fully contained in [offset, offset+length).
    Containment (not overlap) is required: a span straddling a sentence boundary
    is a mis-alignment, and silently half-claiming it would corrupt the
    sentence-relative offsets this returns.

    Three mention classes are treated specially:

      * EXPLETIVE mentions are dropped outright — no key, no span, so the
        sentence is not coref-resolved at all. Coref does link a dummy "It" into
        a company chain ("It's no coincidence that both of these models have
        starting prices under $50,000."), and both consequences are wrong: the
        sentence is not about the company, and injecting the name over the
        pronoun produced "Tesla's no coincidence that ...". See
        _is_expletive_mention().
      * PERSON-LIKE mentions still KEY the sentence (they are legitimate members
        of a resolved chain — see map_coref_clusters()) but are never chosen as
        the span, because that span is what the company name is substituted
        over: "It also claims Mr Musk had himself suggested ..." must not become
        "It also claims Tesla had himself suggested ...". When a sentence's only
        mentions are person-like the span is None, and the ABSA path then scores
        the sentence unsubstituted.
      * NON-SUBSTITUTABLE mentions (is_substitutable_anaphor() is False) are
        given exactly the same treatment as person-like mentions: they still
        KEY the sentence — a coref cluster that legitimately resolves to the
        company by some OTHER mention must still tag this sentence — but are
        never chosen as the span. This is strictly better than filtering only
        in absa.py after the fact: a sentence containing both a junk mention
        ("That list is Australia, Brazil, India, Korea, and Vietnam.") and a
        good pronoun elsewhere now selects the pronoun instead of the junk
        surface, rather than falling back to "no span at all" once the junk
        span is later rejected downstream.

    When a sentence's only mentions are person-like and/or non-substitutable,
    `best` is None and the ABSA path scores the sentence unsubstituted.
    """
    keys: set[str] = set()
    best: tuple[int, int] | None = None
    for start, end, key in coref_mentions:
        if start < offset or end > offset + length or end <= start:
            continue
        rel = (start - offset, end - offset)
        if doc is not None and _is_expletive_mention(doc, start, end, text, rel[0]):
            continue
        keys.add(key)
        surface = text[rel[0] : rel[1]]
        if _is_person_like(surface, start, end, person_spans):
            continue
        if not is_substitutable_anaphor(surface):
            continue
        if best is None or rel[0] < best[0]:
            best = rel
    return keys, best


def tag_sentences(
    article_id,
    sentences,
    ticker: str,
    nlp=None,
    coref_mentions=None,
    use_anaphora_fallback: bool = USE_ANAPHORA_FALLBACK,
) -> pd.DataFrame:
    """Tag one article's already-split sentences with target/other/ceo/subject/
    anaphora flags. See resolve_anaphora() for the tagging algorithm.

    `sentences` accepts EITHER a list of plain strings or a list of spaCy Spans
    (as produced by split_sentences(..., return_spans=True)). Strings are
    parsed on demand — with `nlp` if given, otherwise the shared pipeline from
    _get_nlp(). The point of accepting both is that there is exactly ONE
    tagging implementation: the batch pipeline (process_articles(), which
    parses once and reuses the Spans) and the single-article live path
    (sentiment.analyze(), which has only strings) produce identical tags for
    identical text. A separate string-only fast path would be train/serve skew
    by construction — the demo would disagree with the corpus the model was
    trained on — which is exactly what this module's single-code-path discipline
    exists to prevent.

    Design decision (mentions_ceo vs mentions_target): person-tier aliases
    (e.g. "Musk") are ambiguous — a sentence mentioning Musk may be about
    SpaceX or X, not Tesla. mentions_ceo is therefore tracked as an
    independent, informational flag and never sets mentions_target on its
    own. mentions_target is only set by a "names" alias match or by anaphora
    resolution to a previously-named target. This keeps the ambiguous-person
    signal available for downstream analysis without silently inflating the
    target-sentiment set.

    `coref_mentions` is optional and only meaningful on the Span path, where the
    offsets it carries address the same article-level string the Spans came from
    (see resolve_anaphora()). Omitting it — which every string caller does — runs
    exactly the pre-coref behaviour.
    """
    return resolve_anaphora(
        article_id,
        sentences,
        ticker,
        nlp=nlp,
        coref_mentions=coref_mentions,
        use_anaphora_fallback=use_anaphora_fallback,
    )


def flag_boilerplate(
    sentences_df: pd.DataFrame, min_articles: int = BOILERPLATE_MIN_ARTICLES
) -> pd.DataFrame:
    """Set is_boilerplate=True on sentences whose EXACT text appears in at
    least `min_articles` distinct article_ids.

    Why: plenty of boilerplate survives the MIN_SENT_CHARS length filter —
    disclosure notices, "Image source: Getty Images.", "We've received your
    inquiry and will respond shortly." FinBERT happily scores those like real
    content, and because they repeat across hundreds of articles they drag
    every article-level aggregate toward the same fixed value. Exact-text
    corpus frequency is a cheap, high-precision detector: genuine reporting
    almost never reproduces a full sentence verbatim across five unrelated
    articles, whereas syndicated boilerplate always does.

    This is inherently corpus-level and therefore cannot be computed from a
    single article, so the single-article paths (tag_sentences(),
    sentiment.analyze()) leave is_boilerplate False.

    Returns a copy; the input frame is not mutated.
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

    A cluster refers to a company when ANY of its mention surface forms matches
    that company's registry alias pattern — the target's "names" tier gives the
    key "TARGET", every other COMPANIES entry gives its own key. Only the
    registry is consulted: NER surface forms are not canonical enough to anchor a
    whole coreference chain, and the registry is the same authority the explicit
    path uses, so the two cannot disagree about what a name means.

    A cluster matching MORE THAN ONE company is discarded, not guessed at. Those
    are usually a genuine model error (two firms merged into one chain) or a
    conjoined mention ("both Tesla and Ford ... they"), and either way every
    sentence it would tag is a coin flip. They are counted instead
    (stats["ambiguous_clusters"]) so the cost of that choice stays visible.

    PERSON MENTIONS NEVER KEY A CLUSTER (`person_spans`). A coreference chain
    for a PERSON routinely contains an appositive mention whose surface form
    embeds a company name — "Tesla CEO Elon Musk", "Nvidia's Jensen Huang". A
    plain substring test on that surface keys the WHOLE person-chain to the
    company, so every "he"/"Musk" sentence in the document becomes a target
    sentence. Adjudicating the coref-vs-heuristic disagreement set found this in
    the wild: article 140122179 is a story about Musk's lawsuit against Sam
    Altman, nothing to do with Tesla, and sent_idx=19 — "Kalshi's similar market
    has Musk winning at 55%." — was tagged mentions_target purely because the
    Musk chain contained "Tesla CEO Elon Musk".

    That silently bypassed a deliberate design decision that predates coref:
    mentions_target has NEVER been settable by a person-tier match, because a
    Musk sentence may be about SpaceX or X rather than Tesla (see
    tag_sentences()). Coref must not be a back door into it.

    So `person_spans` — (start_char, end_char) of every PERSON entity in the
    same spaCy doc, in the same coordinate system as the cluster spans — marks
    mentions that are UNUSABLE FOR COMPANY KEYING. process_articles() threads
    the parsed doc's PERSON entities through; the argument defaults to None
    (= no filtering) so unit tests and any caller without a parse behave as
    before. This is one of the three PERSON consumers that keep spaCy's `ner`
    component load-bearing after the ORG detector's removal — see _get_nlp().

    TITLE NOUN PHRASES NEVER KEY A CLUSTER EITHER. The PERSON-overlap rule above
    only catches mentions that literally contain a person's name. It misses the
    other half of the same phenomenon: an appositive TITLE noun phrase that names
    a person purely by their company — "the Tesla CEO", "The Tesla Inc. (NASDAQ:
    TSLA) CEO", "Tesla's co-founder". These contain NO PERSON entity at all, so
    they slip past `person_spans`, yet they are exactly as person-referring as
    "Tesla CEO Elon Musk" is: the mention denotes a human being, and keying his
    coreference chain to the company is precisely what the person-tier design
    forbids — mentions_target must never be set by a person-tier match alone (see
    tag_sentences()). A mention containing any of the job-title tokens in
    _TITLE_TOKEN_RE (CEO, chief executive, CFO, CTO, COO, founder, co-founder,
    chairman, chairwoman, president, boss) is therefore also unusable for keying.

    Measured on the corpus this removes ~174 mentions_target sentences. Concrete
    cases it fixes: article 140122179 sent_idx 19 (the Musk-vs-Altman lawsuit
    story, where "the Tesla CEO" keyed the whole Musk chain to Tesla), and
    article 141052996 sent_idx 8 and 26.

    THIS AFFECTS THE KEYING DECISION ONLY. A person-like mention still belongs
    to its cluster: if some OTHER, non-person mention in the same chain names a
    company ("Tesla ... the company ... Tesla CEO Elon Musk ... it"), the
    cluster still resolves and ALL of its mentions — the person-like one
    included — still tag their sentences exactly as before.

    Returns (mentions, stats): mentions is a flat list of
    (abs_char_start, abs_char_end, key) for every mention of every
    unambiguously-resolved cluster, and stats counts clusters seen, resolved,
    ambiguous, and unresolved (no company named anywhere in the chain).
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
    use_anaphora_fallback: bool = USE_ANAPHORA_FALLBACK,
) -> pd.DataFrame:
    """Batch entry point: sentence-split df[text_col] via nlp.pipe, tag each
    article's sentences, concatenate into one long DataFrame, and run the
    corpus-level flag_boilerplate() pass over the result (which can only be
    done here, once all articles are in one frame).

    Uses the Span path end-to-end (split_sentences(..., return_spans=True)) so
    the corpus is parsed exactly once: the dependency parse and the named
    entities that tagging needs are the same ones that produced the sentence
    boundaries, and re-parsing per sentence would roughly double the cost.

    COREFERENCE. With use_coref=True (config.USE_COREF) and a working backend,
    the article bodies are run through coref.resolve_documents() first and the
    resulting clusters are mapped onto companies (map_coref_clusters()); the
    per-article mentions are handed to tag_sentences(), where they fill the same
    slot the anaphora heuristic fills, at higher precedence. If the backend is
    unavailable the pipeline logs one warning and runs the heuristic alone —
    coref is never allowed to break the run. resolve_documents() is transparently
    cached on disk (config.COREF_CACHE_PATH), so a re-run over an already-seen
    corpus skips inference entirely.

    The parse's PERSON entities are passed to map_coref_clusters() so that a
    person-chain cannot key itself to a company through a mention like "Tesla
    CEO Elon Musk" — see that function's docstring.

    ALIGNMENT. Coref MUST see the exact string the sentence Spans were parsed
    from, or its character offsets address different characters than the
    sentence boundaries do. split_sentences() applies _fix_missing_space() to
    every body before spaCy, so the cleaned text — not df[text_col] — is what is
    sent to coref, and the identity is then asserted against the parsed Doc
    below rather than assumed.

    `use_anaphora_fallback` (default config.USE_ANAPHORA_FALLBACK = False) is
    threaded straight through to resolve_anaphora() -- see its docstring for
    the measured 13%-vs-74% precision evidence behind the default.

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
            logger.warning(
                "Coreference requested but unavailable; falling back to the anaphora heuristic"
            )
        else:
            all_clusters = coref.resolve_documents(cleaned)
            patterns = _build_ticker_patterns(ticker)["names"]
            totals = {"clusters": 0, "resolved": 0, "ambiguous": 0, "unresolved": 0}
            n_docs_without_usable = 0
            n_misaligned = 0
            for i, (clusters, sentences) in enumerate(zip(all_clusters, all_sentences)):
                # Explicit alignment check: the Doc the Spans came from must be
                # character-identical to the string coref was given, otherwise
                # every offset below points at the wrong characters.
                if sentences and sentences[0].doc.text != cleaned[i]:
                    n_misaligned += 1
                    continue
                # PERSON entities from the SAME parse the Spans came from, in
                # the same character coordinates as the cluster spans. They
                # block company keying off mentions like "Tesla CEO Elon Musk"
                # — see map_coref_clusters()'s docstring.
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
                    "those fall back to the heuristic"
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
                use_anaphora_fallback=use_anaphora_fallback,
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
