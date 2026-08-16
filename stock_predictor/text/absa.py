"""Aspect-based sentiment (ABSA) over entity-tagged sentences.

WHY THIS EXISTS. FinBERT scores a SENTENCE. The pipeline asks a different
question: how does this sentence read ABOUT THE TARGET COMPANY? Those come apart
exactly where it hurts most. "Build-A-Bear outperformed Tesla" is an
unambiguously positive sentence and FinBERT scores it positive; credited to
Tesla it is backwards. sentiment.aggregate_article_features() already carries a
sent_entity_excl_comp_* mean precisely because notebook 2.0 measured this
pollution and could only respond by DELETING the offending sentences. An
aspect-based model is handed the aspect term explicitly and scores sentiment
toward it, so the comparative sentence can be kept and read correctly instead.

THIS RUNS ALONGSIDE FinBERT, NOT INSTEAD OF IT. Every column this module feeds
into sentiment.aggregate_article_features() is a NEW absa_* column parallel to
an existing sent_* one; no FinBERT column changes name, semantics or value. The
model here (config.ABSA_MODEL) is SemEval/review-trained, not finance-trained,
so the plausible failure mode is "better on the ~3% of comparative sentences,
worse on the other 97%". That is a measurement to run, not an assumption to
make, which is why both families of features coexist.

MODULE SHAPE mirrors sentiment.py and coref.py deliberately:
    is_available()        -> bool; the backend imported and the model loaded
    score_pairs()         -> core batched + cached (text, aspect) scorer
    build_pairs()         -> sentence table -> absa_text / absa_aspect columns
    score_sentence_table()-> build_pairs() + score_pairs(), merged back on
    load_cache()/save_cache() -> on-disk parquet cache, MERGE never replace

CACHE KEY. sentiment.hash_text(text + "\\x00" + aspect) — the SAME hashing
implementation as FinBERT's cache, deliberately not a second copy, over text and
aspect JOINTLY. The aspect is half the input to the model, so a key over text
alone would serve a Tesla-aspect score for a Ford-aspect query. NUL is used as
the separator because it cannot occur in the corpus text, so no (text, aspect)
pair can collide with a different pair by rearranging the boundary.

MERGE-NOT-REPLACE. Every save concatenates with the cache loaded at the start of
the call before deduping. sentiment.py's module docstring records the corpus-run
truncation bug that came from writing only the current call's rows; the same
shape of bug here would throw away a full ABSA pass, so the pattern is repeated
rather than re-derived.

GRACEFUL DEGRADATION. Nothing in this module may hard-fail the pipeline. A
missing/broken model means one warning and NaN scores; a corrupt cache means one
warning and a cold run; an unusable substitution span means the unchanged text.
"""

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from stock_predictor.config import (
    ABSA_BATCH_SIZE,
    ABSA_CACHE_PATH,
    ABSA_MODEL,
    COMPANIES,
    MAX_TOKENS,
)
from stock_predictor.text import entity_filter, sentiment

CACHE_COLUMNS = ["pair_hash", "pos", "neg", "neu"]

# Columns build_pairs() writes onto the sentence table.
PAIR_COLUMNS = ["absa_text", "absa_aspect"]

# Columns score_sentence_table() writes onto the sentence table. Named apart
# from FinBERT's pos/neg/neu so a table can carry both models at once.
SCORE_COLUMNS = ["absa_pos", "absa_neg", "absa_neu"]

# Cached model handle, mirroring sentiment.py's lazy load and coref._MODEL.
# _LOAD_FAILED latches so a missing backend warns exactly once per process.
_MODEL = None
_TOKENIZER = None
_LOAD_FAILED = False


def pair_hash(text: str, aspect: str) -> str:
    """Cache key for one (text, aspect) pair.

    Delegates to sentiment.hash_text() over the NUL-joined pair so that there is
    one hashing implementation in the codebase, and so that text and aspect
    TOGETHER form the key — see the module docstring.
    """
    return sentiment.hash_text(f"{(text or '').strip()}\x00{(aspect or '').strip()}")


def load_cache(path=ABSA_CACHE_PATH) -> pd.DataFrame:
    """Load the on-disk ABSA score cache.

    Returns an empty frame with CACHE_COLUMNS when the file does not exist.
    NEVER raises: a corrupt, truncated or schema-drifted file logs one warning
    and is treated as empty, so the run pays for a cold pass instead of dying on
    a cache it could have ignored. (Same laxer policy as coref.load_cache(): the
    cache is a pure speed optimisation and cannot change any output value.)
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
            f"ABSA cache at {path} is unreadable "
            f"({type(exc).__name__}: {exc}); treating it as empty"
        )
        return pd.DataFrame(columns=CACHE_COLUMNS)


def save_cache(cache_df: pd.DataFrame, path=ABSA_CACHE_PATH) -> None:
    """Persist the cache, deduped on pair_hash (keep last).

    Callers must pass the MERGE of the cache they loaded and the rows they
    produced, never the new rows alone — see the module docstring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped = cache_df.drop_duplicates(subset="pair_hash", keep="last")[CACHE_COLUMNS]
    deduped.to_parquet(path, index=False)
    logger.info(f"Saved ABSA cache with {len(deduped)} entries to {path}")


def _load_model_and_tokenizer():
    """Load (once, then memoise) the ABSA model. Returns (None, None) on failure.

    The model is a standard AutoModelForSequenceClassification used in its
    SENTENCE-PAIR form: tokenizer(text, aspect) -> 3 logits over
    negative/neutral/positive. Its label order is read from config.id2label at
    load time by _build_label_index_map(), never assumed — this checkpoint
    happens to be {0: Negative, 1: Neutral, 2: Positive}, which is NOT FinBERT's
    order, so an index assumption copied from sentiment.py would have silently
    swapped positive and negative.
    """
    global _MODEL, _TOKENIZER, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL, _TOKENIZER
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading ABSA model '{ABSA_MODEL}'")
        _TOKENIZER = AutoTokenizer.from_pretrained(ABSA_MODEL)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL)
        _MODEL.eval()
        logger.info(f"ABSA model loaded; id2label={_MODEL.config.id2label}")
    except Exception as exc:  # noqa: BLE001 - any failure means "degrade"
        _LOAD_FAILED = True
        _MODEL = None
        _TOKENIZER = None
        logger.warning(
            f"ABSA backend unavailable ({type(exc).__name__}: {exc}); "
            "aspect-based scores will be NaN"
        )
    return _MODEL, _TOKENIZER


def is_available() -> bool:
    """True when the ABSA backend imported AND the model loaded successfully.

    Callers use this as the enable-gate; a False here must always mean "skip
    ABSA and leave the columns NaN", never "raise".
    """
    return _load_model_and_tokenizer()[0] is not None


def score_pairs(
    pairs: list[tuple[str, str]],
    model=None,
    tokenizer=None,
    cache_df: pd.DataFrame | None = None,
    batch_size: int = ABSA_BATCH_SIZE,
) -> pd.DataFrame:
    """Core batched + cached aspect-based scorer.

    `pairs` is a list of (text, aspect). Structurally this is
    sentiment.score_sentences() with a two-part key and a two-part model input:

      * every pair is hashed with pair_hash() and DEDUPED, because the same
        (sentence, aspect) recurs across articles (syndicated copy) and within
        one article's repeated framing;
      * hashes present in `cache_df` are served from it and never re-scored;
      * the needs-scoring remainder is sorted by combined length before batching
        so same-batch sequences pad to similar lengths;
      * batches run under torch.no_grad(), truncated to MAX_TOKENS;
      * label columns are mapped BY NAME through
        sentiment._build_label_index_map() reading model.config.id2label, never
        by hardcoded index — this checkpoint's order differs from FinBERT's.

    If every hash is a cache hit the model is NEVER LOADED, even with
    model/tokenizer left None. That makes warm runs cheap and lets tests score
    without a model at all.

    Degradation: if the model cannot be loaded, the un-cached pairs are simply
    omitted from the result (one warning). Callers left-merge, so those rows end
    up NaN rather than raising.

    Returns one row per UNIQUE pair_hash, columns pair_hash, pos, neg, neu.
    Does NOT write to disk — callers own when to call save_cache().
    """
    if cache_df is None or len(cache_df) == 0:
        cache_lookup: dict[str, tuple[float, float, float]] = {}
    else:
        cache_lookup = {
            row.pair_hash: (row.pos, row.neg, row.neu)
            for row in cache_df.itertuples(index=False)
        }

    unique_by_hash: dict[str, tuple[str, str]] = {}
    for text, aspect in pairs:
        h = pair_hash(text, aspect)
        if h not in unique_by_hash:
            unique_by_hash[h] = (text, aspect)

    cached_rows = []
    to_score: list[tuple[str, str, str]] = []  # (hash, text, aspect)
    for h, (text, aspect) in unique_by_hash.items():
        if h in cache_lookup:
            pos, neg, neu = cache_lookup[h]
            cached_rows.append({"pair_hash": h, "pos": pos, "neg": neg, "neu": neu})
        else:
            to_score.append((h, text, aspect))

    logger.info(
        f"score_pairs: {len(unique_by_hash)} unique (text, aspect) pairs "
        f"({len(cached_rows)} cache hits, {len(to_score)} to score)"
    )

    if not to_score:
        return pd.DataFrame(cached_rows, columns=CACHE_COLUMNS)

    if model is None or tokenizer is None:
        loaded_model, loaded_tokenizer = _load_model_and_tokenizer()
        model = model or loaded_model
        tokenizer = tokenizer or loaded_tokenizer

    if model is None or tokenizer is None:
        logger.warning(
            f"score_pairs: no ABSA model available; {len(to_score)} pairs left unscored"
        )
        return pd.DataFrame(cached_rows, columns=CACHE_COLUMNS)

    label_idx = sentiment._build_label_index_map(model)
    pos_i, neg_i, neu_i = label_idx["positive"], label_idx["negative"], label_idx["neutral"]

    # Sort by combined character length (cheap proxy for token length) so that
    # same-batch sequences pad to similar lengths.
    to_score.sort(key=lambda triple: len(triple[1]) + len(triple[2]))

    scored_rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(to_score), batch_size), desc="Scoring pairs (ABSA)"):
            batch = to_score[start : start + batch_size]
            batch_hashes = [h for h, _, _ in batch]
            batch_texts = [t for _, t, _ in batch]
            batch_aspects = [a for _, _, a in batch]

            inputs = tokenizer(
                batch_texts,
                batch_aspects,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

            for h, row in zip(batch_hashes, probs.tolist()):
                scored_rows.append(
                    {
                        "pair_hash": h,
                        "pos": row[pos_i],
                        "neg": row[neg_i],
                        "neu": row[neu_i],
                    }
                )

    return pd.DataFrame(cached_rows + scored_rows, columns=CACHE_COLUMNS)


def _display_name(key: str) -> str | None:
    """The canonical display name of a COMPANIES entry: its first "names" alias.

    First rather than longest/shortest because the registry lists the natural
    written form first ("Tesla", not "Tesla, Inc." or "$TSLA"), which is what an
    ABSA model trained on prose expects an aspect term to look like.
    """
    names = COMPANIES.get(key, {}).get("names", []) or []
    return names[0] if names else None


# Pronoun surface forms that must inflect the injected name rather than be
# replaced by it verbatim — see _inflect_name().
_POSSESSIVE_PRONOUNS = frozenset({"its", "their", "his", "her", "our", "your"})
_CONTRACTIONS_IS = frozenset({"it's", "it’s", "he's", "he’s", "she's", "she’s"})
_CONTRACTIONS_ARE = frozenset({"they're", "they’re", "we're", "we’re"})


def _inflect_name(name: str, surface: str) -> str:
    """Inflect `name` to fit grammatically where `surface` stood.

    Substitution replaces the anaphor's own characters with the company name,
    and a bare name does not fit every anaphor:

        "Its revenue increased 16%..."          -> "Tesla revenue increased..."
        "This is a direct result of its price cuts" -> "of Tesla price cuts"

    Both are ungrammatical, and an ABSA model reading them is being handed
    broken input on a sentence that was otherwise fine. So the surface form is
    inspected and the injected name inflected to match:

      * a possessive pronoun ("its", "their", "his", "her") or a surface already
        ending in "'s"/"’s" ("the company's") -> "<Name>'s";
      * a subject+copula contraction ("it's", "he's") -> "<Name> is", and
        ("they're", "we're") -> "<Name> are" — genuinely referential ones only,
        since expletive "It's" is dropped upstream (see
        entity_filter._is_expletive_mention());
      * anything else -> the bare name, which is the original behaviour.

    Capitalisation follows the surface: a surface that was capitalised (a
    sentence-initial "Its") yields a capitalised injection. Registry display
    names are proper nouns and already capitalised, so this only matters for a
    lowercase display name.
    """
    stripped = (surface or "").strip()
    low = stripped.lower()
    if low in _CONTRACTIONS_ARE:
        out = f"{name} are"
    elif low in _CONTRACTIONS_IS:
        out = f"{name} is"
    elif low in _POSSESSIVE_PRONOUNS or low.endswith("'s") or low.endswith("’s"):
        base = name
        if not (base.lower().endswith("'s") or base.lower().endswith("’s")):
            base = f"{base}'s"
        out = base
    else:
        out = name
    if stripped[:1].isupper() and out[:1].islower():
        out = out[:1].upper() + out[1:]
    return out


def _substitute(text: str, start, end, name: str) -> tuple[str, bool]:
    """Replace text[start:end] with `name`, INFLECTED to fit the surface form it
    replaces (see _inflect_name()). Returns (new_text, substituted).

    Any unusable span — NA, non-integer, negative, reversed, past the end of the
    string — falls back to the UNCHANGED text and returns False. It never
    raises: a bad offset is a data-quality event on one sentence, and the caller
    logs it at debug and scores the sentence unsubstituted rather than losing
    the row or the run.
    """
    try:
        if start is None or end is None or pd.isna(start) or pd.isna(end):
            return text, False
        s, e = int(start), int(end)
    except (TypeError, ValueError):
        return text, False
    if s < 0 or e > len(text) or e <= s:
        return text, False
    return text[:s] + _inflect_name(name, text[s:e]) + text[e:], True


def _substitute_resolved(text: str, start, end, name: str) -> tuple[str, str]:
    """Substitute `name` into `text` over the resolved anaphor span, if allowed.

    Returns (text_to_score, outcome), outcome being one of:
      "substituted"  — the name was injected (inflected by _substitute());
      "person"       — the span is a PERSON mention, so the sentence is scored
                       UNCHANGED. A person mention must never be spoken for by
                       the company: "It also claims Mr Musk had himself
                       suggested turning OpenAI into a commercial venture."
                       must not become "It also claims Tesla had himself
                       suggested ...". The test is
                       entity_filter._is_person_like(), the same predicate that
                       forbids a person mention from keying a coref cluster;
                       here it is the surface-only form, since this module
                       holds the sentence text but not the parse
                       (entity_filter applies the PERSON-entity half of the
                       same predicate upstream and simply records no span for
                       such a mention);
      "not_anaphor"  — the span is present and not a PERSON mention, but is
                       not a recognisably company-anaphoric surface either
                       (entity_filter.is_substitutable_anaphor() is False), so
                       the sentence is scored UNCHANGED. This is
                       DEFENCE-IN-DEPTH: entity_filter already refuses to
                       CHOOSE a non-substitutable span when a substitutable one
                       is available in the same sentence (see
                       _coref_hits_in_sentence()), but a sentence whose ONLY
                       coref mention is non-substitutable still records that
                       span, and the recency-heuristic anaphor path (generic
                       "the company" / descriptor matching) has never been
                       filtered by this predicate at all. Real corpus
                       breakage this guards against: "Both stocks trade at
                       steep valuations" -> "Tesla trade at steep valuations",
                       and "It was a remarkable moment in the earnings call."
                       -> "...a remarkable moment in Tesla.";
      "no_span"      — the row is flagged resolved but carries no span at all;
      "bad_span"     — the span is present but unusable (reversed, out of
                       range).

    All non-"substituted" outcomes score the UNCHANGED text, which leaves the
    model with no aspect anchor — the same position FinBERT is in — rather
    than with nonsense.
    """
    try:
        missing = start is None or end is None or pd.isna(start) or pd.isna(end)
    except TypeError:
        missing = True
    if missing:
        return text, "no_span"
    try:
        surface = text[int(start) : int(end)]
    except (TypeError, ValueError):
        surface = ""
    if surface and entity_filter._is_person_like(surface):
        return text, "person"
    if not entity_filter.is_substitutable_anaphor(surface):
        return text, "not_anaphor"
    new_text, ok = _substitute(text, start, end, name)
    if not ok:
        return text, "bad_span"
    return new_text, "substituted"


def _other_aspect(key, source, text: str, other_patterns: dict) -> str | None:
    """The aspect term for a row tagged mentions_other, or None if unrecoverable.

    Registry-sourced keys map to the registry display name, as they always have.
    NER-sourced keys have no registry entry — they ARE the normalized surface
    form entity_filter recorded ("avidity biosciences") — so the surface itself
    is the aspect, title-cased for readability. Before this, _display_name()
    returned None for every one of them and the row was dropped: 11,370 of the
    23,257 mentions_other rows on the corpus had no aspect at all, which is the
    entire reason absa_other_mean_* was so much thinner than sent_other_mean_*.

    Rows whose table predates the other_key column (or that were built by hand)
    fall back to re-deriving a registry key from the text, the old behaviour.
    """
    key = (key or "").strip() if isinstance(key, str) else ""
    if key and source == "ner":
        return key.title()
    if key:
        name = _display_name(key)
        if name:
            return name
    fallback = _earliest_other_key(text, other_patterns)
    return _display_name(fallback) if fallback else None


def build_pairs(sentences_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Turn a TAGGED sentence table into (text, aspect) scoring pairs.

    Adds exactly two columns and modifies nothing else — in particular the
    original `text` column is never rewritten:

        absa_text   : the string actually handed to the model
        absa_aspect : the aspect term handed to the model alongside it

    WHY SUBSTITUTION EXISTS (the subtle part). An ABSA model attends to an
    aspect TERM that must be PRESENT IN THE TEXT; it is not a classifier over an
    abstract entity id. An anaphoric sentence — "The company also raised prices
    across its lineup." — contains no company name for the model to attend to,
    so passing aspect="Tesla" asks it to score a term that does not occur. So we
    INJECT the resolved name over the anaphor's own characters, using the
    anaphor_char_start/end spans entity_filter records for exactly this purpose
    (coref.py's module docstring calls this out as the downstream reason those
    spans are character offsets rather than token indices):

        "The company also raised prices."  ->  "Tesla also raised prices."

    THIS MAKES ABSA QUALITY DEPENDENT ON ANTECEDENT QUALITY. A wrong antecedent
    does not merely mis-file a correct score any more; it changes the sentence
    the model reads. That dependency is the reason coreference landed before
    this module rather than after it: roughly a third of all target sentences
    arrive through anaphora, and the recency heuristic hand-scored at ~60%.

    RULES, in priority order (one pair per row — see the note below):
      1. Rows where sentiment.needs_score() is False get empty absa_text and
         absa_aspect and are not scored. Same predicate as FinBERT uses, so the
         two scorers cover the same rows.
      2. mentions_target, named explicitly (neither resolved_by_coref nor
         resolved_by_anaphora): absa_text = text UNCHANGED, aspect = the
         target's display name. The name is already in the sentence.
      3. mentions_target, resolved (resolved_by_coref OR resolved_by_anaphora)
         with a usable span: substitute the target's display name over the
         span; aspect = that display name. The injected name is INFLECTED to
         the surface it replaces ("its" -> "Tesla's") — see _inflect_name(). A
         span that is a PERSON mention is never substituted over, NOR is a span
         that is not a recognisably company-anaphoric surface at all
         (entity_filter.is_substitutable_anaphor() is False — e.g. a plural
         "both companies", a demonstrative alone, a proper noun a coref error
         dragged into the chain, or an arbitrary long NP that merely ends in a
         company word); an unusable/absent span falls back to the unchanged
         text too (logged at debug); see _substitute_resolved(). All of these
         leave the aspect absent from the text — the model then reads the
         sentence with no aspect anchor, which is the same position FinBERT is
         in, so it degrades to sentence sentiment rather than to nonsense.
      4. mentions_other and NOT mentions_target: aspect = that company's
         display name (registry) or its title-cased NER surface form, from the
         other_key/other_source columns — see _other_aspect(); substituted the
         same way when the row was resolved rather than named.
      5. mentions_ceo and NOT mentions_target: aspect = the matched person
         alias as it appears in the sentence ("Musk", "Elon Musk"), text
         unchanged. The company is deliberately NOT substituted here: Musk is a
         legitimate aspect in his own right, and the person tier has never been
         allowed to speak for the company (see entity_filter.tag_sentences()).

    ONE PAIR PER ROW. A comparative sentence is both a target row and an other
    row, but carries a single absa_aspect, and rule 2/3 gives it the TARGET
    aspect — which is the right choice, since reading comparatives correctly for
    the target is the entire motivation for this module. The consequence is that
    such a row contributes its TARGET-aspect score to absa_other_mean_*, where
    FinBERT contributes one sentence-level score to both. Documented rather than
    hidden; absa_entity_excl_comp_* drops these rows anyway.

    KNOWN GAP (narrowed). Rule 4 needs the other company's identity. The
    sentence table now records it directly (entity_filter's `other_key`), so
    both registry-named and NER-named other companies get an aspect. What
    remains unrecoverable is an other company reached by COREF/ANAPHORA: those
    rows name no company and carry no key, so they get no pair and end up NaN.
    This still narrows the absa_other_mean_* population relative to
    sent_other_mean_*; it does not bias the target features at all.

    Returns a COPY; the input frame is not mutated.
    """
    out = sentences_df.copy()
    if len(out) == 0:
        out["absa_text"] = pd.Series(dtype=object)
        out["absa_aspect"] = pd.Series(dtype=object)
        return out

    mask = sentiment.needs_score(out)
    target_name = _display_name(ticker)
    other_patterns = entity_filter._build_other_company_patterns(ticker)
    person_pattern = entity_filter._build_ticker_patterns(ticker)["person"]

    # Default: no pair. Rows the rules below never touch stay empty, which is
    # what "not scored" looks like on this table.
    texts: list[str] = [""] * len(out)
    aspects: list[str] = [""] * len(out)
    outcomes = {
        "substituted": 0,
        "person": 0,
        "not_anaphor": 0,
        "no_span": 0,
        "bad_span": 0,
    }
    n_unresolved_other = 0

    # Positional iteration: `mask` is index-aligned to `out`, so take its values.
    mask_values = mask.to_numpy()
    text_values = out["text"].tolist()
    m_target = out["mentions_target"].fillna(False).astype(bool).tolist()
    m_other = out["mentions_other"].fillna(False).astype(bool).tolist()
    m_ceo = out["mentions_ceo"].fillna(False).astype(bool).tolist()
    resolved = (
        out.get("resolved_by_coref", pd.Series(False, index=out.index))
        .fillna(False)
        .astype(bool)
        | out.get("resolved_by_anaphora", pd.Series(False, index=out.index))
        .fillna(False)
        .astype(bool)
    ).tolist()
    span_start = out.get("anaphor_char_start", pd.Series([None] * len(out))).tolist()
    span_end = out.get("anaphor_char_end", pd.Series([None] * len(out))).tolist()
    other_source = (
        out.get("other_source", pd.Series([""] * len(out), index=out.index))
        .fillna("")
        .tolist()
    )
    other_key = (
        out.get("other_key", pd.Series([""] * len(out), index=out.index)).fillna("").tolist()
    )

    for i in range(len(out)):
        if not mask_values[i]:
            continue
        text = text_values[i] or ""

        if m_target[i]:
            name = target_name
            if name is None:
                continue
            if resolved[i]:
                new_text, outcome = _substitute_resolved(
                    text, span_start[i], span_end[i], name
                )
                outcomes[outcome] += 1
                if outcome == "bad_span":
                    logger.debug(
                        f"build_pairs: unusable anaphor span "
                        f"({span_start[i]}, {span_end[i]}) on a {len(text)}-char sentence; "
                        "scoring the unchanged text"
                    )
                texts[i], aspects[i] = new_text, name
            else:
                texts[i], aspects[i] = text, name
            continue

        if m_other[i]:
            name = _other_aspect(other_key[i], other_source[i], text, other_patterns)
            if name is None:
                n_unresolved_other += 1
                continue
            if resolved[i]:
                new_text, outcome = _substitute_resolved(
                    text, span_start[i], span_end[i], name
                )
                outcomes[outcome] += 1
                texts[i], aspects[i] = new_text, name
            else:
                texts[i], aspects[i] = text, name
            continue

        if m_ceo[i] and person_pattern is not None:
            m = person_pattern.search(text)
            if m is None:
                continue
            texts[i], aspects[i] = text, m.group(0)

    out["absa_text"] = texts
    out["absa_aspect"] = aspects

    n_pairs = int(sum(1 for a in aspects if a))
    logger.info(
        f"build_pairs: {n_pairs} pairs built from {int(mask_values.sum())} scoreable rows "
        f"({len(out)} total); {n_unresolved_other} other-company rows had no recoverable "
        f"aspect; resolved rows: {outcomes['substituted']} substituted, "
        f"{outcomes['person']} left unchanged (person mention), "
        f"{outcomes['not_anaphor']} left unchanged (not a company anaphor), "
        f"{outcomes['no_span']} with no span recorded, "
        f"{outcomes['bad_span']} with an unusable span"
    )
    return out


def _earliest_other_key(text: str, other_patterns: dict) -> str | None:
    """Registry key of the non-target company named EARLIEST in `text`, or None.

    Earliest-mention is the same fallback precedence entity_filter uses when it
    has no better signal (see resolve_anaphora()); reusing it keeps the aspect we
    pick consistent with the antecedent the tagger would have picked.
    """
    best_key = None
    best_pos = None
    for key, pat in other_patterns.items():
        m = pat.search(text)
        if m is None:
            continue
        if best_pos is None or m.start() < best_pos:
            best_key, best_pos = key, m.start()
    return best_key


def score_sentence_table(
    sentences_df: pd.DataFrame,
    ticker: str,
    cache_path=ABSA_CACHE_PATH,
) -> pd.DataFrame:
    """Build pairs for `sentences_df` and score them, returning the table with
    absa_text, absa_aspect, absa_pos, absa_neg, absa_neu attached.

    The FinBERT counterpart of this function is
    sentiment.score_sentence_table(), and the same three decisions are made the
    same way, for the same reasons:
      * only rows sentiment.needs_score() marks are scored (via build_pairs());
      * rows without a pair get NaN, never 0 — 0.0 is a real probability the
        model could have returned, so writing it would conflate "never scored"
        with "scored and confidently not positive";
      * the cache is saved ONCE at the end of the call, MERGED with the cache
        loaded at its start.

    Returns a copy; the input frame is not mutated.
    """
    out = build_pairs(sentences_df, ticker)
    if len(out) == 0:
        for col in SCORE_COLUMNS:
            out[col] = pd.Series(dtype=float)
        return out

    cache_df = load_cache(cache_path)
    pairs = [
        (t, a)
        for t, a in zip(out["absa_text"].tolist(), out["absa_aspect"].tolist())
        if a
    ]
    scored = score_pairs(pairs, cache_df=cache_df)

    # Merge, never replace — see the module docstring.
    save_cache(pd.concat([cache_df, scored], ignore_index=True), path=cache_path)

    keys = [
        pair_hash(t, a) if a else None
        for t, a in zip(out["absa_text"].tolist(), out["absa_aspect"].tolist())
    ]
    out["pair_hash"] = keys
    merged = out.merge(
        scored.rename(columns={"pos": "absa_pos", "neg": "absa_neg", "neu": "absa_neu"}),
        on="pair_hash",
        how="left",
    ).drop(columns=["pair_hash"])
    for col in SCORE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged
