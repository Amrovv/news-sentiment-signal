"""Aspect-based sentiment (ABSA) over entity-tagged sentences.

FinBERT scores a sentence; this scores how a sentence reads about a given aspect,
so a comparative sentence can be attributed correctly rather than dropped. Runs
alongside FinBERT, writing absa_* columns parallel to the sent_* ones.

    is_available()          the backend imported and the model loaded
    build_pairs()           sentence table -> absa_text / absa_aspect
    score_pairs()           batched, cached (text, aspect) scorer
    score_sentence_table()  both of the above, merged back on
    load_cache()/save_cache()

Cache key is hash_text(text + "\x00" + aspect), over both jointly, since the aspect
is half the model input. Saves MERGE with the cache loaded at call start.

Nothing here hard-fails the pipeline: a broken model gives NaN, a corrupt cache a
cold run, an unusable span the unchanged text.
"""

from loguru import logger
import pandas as pd
import torch
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
    """Load the on-disk ABSA score cache, or an empty frame when absent.

    Never raises: a corrupt file logs one warning and reads as empty, costing a cold
    pass. The cache affects speed only, never output values.
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

    Used in its sentence-pair form: tokenizer(text, aspect) -> 3 logits. Label order
    is read from config.id2label, never assumed; this checkpoint is
    {0: Negative, 1: Neutral, 2: Positive}, unlike FinBERT.
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

    `pairs` is a list of (text, aspect). Pairs are deduped on pair_hash, cache hits
    served, and the remainder sorted by combined length before batching so same-batch
    sequences pad alike. Label columns are mapped by name from model.config.id2label;
    this checkpoint's order differs from FinBERT's.

    If every hash hits, the model is never loaded. If it cannot be loaded, un-cached
    pairs are omitted with one warning and callers' left-merge turns them into NaN.

    Returns one row per unique pair_hash. Does not write to disk.
    """
    if cache_df is None or len(cache_df) == 0:
        cache_lookup: dict[str, tuple[float, float, float]] = {}
    else:
        cache_lookup = {
            row.pair_hash: (row.pos, row.neg, row.neu) for row in cache_df.itertuples(index=False)
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

# Bare pronoun surfaces whose span may exclude a trailing clitic; see
# _consume_trailing_clitic().
_BARE_PRONOUN_SURFACES = frozenset({"it", "they", "he", "she"})

# Clitic -> the auxiliary it expands into, always singular since the injected
# name is one company. Longest-first so "'s" cannot match inside "'re". Straight
# and curly apostrophes both appear in scraped text.
_CLITIC_EXPANSIONS = [
    ("'ll", "will"),
    ("’ll", "will"),
    ("'ve", "has"),
    ("’ve", "has"),
    ("'re", "is"),
    ("’re", "is"),
    ("'s", "is"),
    ("’s", "is"),
    ("'d", "would"),
    ("’d", "would"),
]


def _consume_trailing_clitic(text: str, surface: str, end: int) -> tuple[str | None, int]:
    """Detect an apostrophe clitic glued to the end of the replaced span.

    The resolved span often covers only the pronoun ("It" in "It's also profitable"),
    leaving the clitic just outside it, so replacing naively yields the possessive
    "Tesla's also profitable" for a sentence meaning "Tesla is also profitable".

    Returns (expansion, clitic_len) when `surface` is a bare, non-possessive pronoun
    and `text` after `end` begins with a known clitic; otherwise (None, 0). Possessive
    surfaces are handled by _inflect_name() instead.
    """
    low = (surface or "").strip().lower()
    if low not in _BARE_PRONOUN_SURFACES:
        return None, 0
    tail = text[end:]
    for clitic, expansion in _CLITIC_EXPANSIONS:
        if tail.lower().startswith(clitic.lower()):
            return expansion, len(clitic)
    return None, 0


def _inflect_name(name: str, surface: str) -> str:
    """Inflect `name` to fit grammatically where `surface` stood.

    A possessive pronoun or a surface ending in "'s" gives "<Name>'s"; "it's"/"he's"
    gives "<Name> is"; "they're"/"we're" gives "<Name> are"; anything else the bare
    name. Capitalisation follows the surface.
    """
    stripped = (surface or "").strip()
    low = stripped.lower()
    if low in _CONTRACTIONS_ARE:
        out = f"{name} are"
    elif low in _CONTRACTIONS_IS:
        out = f"{name} is"
    elif low in _POSSESSIVE_PRONOUNS or low.endswith(("'s", "’s")):
        base = name
        if not base.lower().endswith(("'s", "’s")):
            base = f"{base}'s"
        out = base
    else:
        out = name
    if stripped[:1].isupper() and out[:1].islower():
        out = out[:1].upper() + out[1:]
    return out


def _substitute(text: str, start, end, name: str) -> tuple[str, bool]:
    """Replace text[start:end] with `name`, inflected to fit the surface it replaces.

    Returns (new_text, substituted). An unusable span falls back to the unchanged text
    and returns False rather than raising.

    A bare pronoun span followed by an apostrophe clitic consumes and expands it, so
    contractions are never split. See _consume_trailing_clitic().
    """
    try:
        if start is None or end is None or pd.isna(start) or pd.isna(end):
            return text, False
        s, e = int(start), int(end)
    except (TypeError, ValueError):
        return text, False
    if s < 0 or e > len(text) or e <= s:
        return text, False
    surface = text[s:e]
    expansion, clitic_len = _consume_trailing_clitic(text, surface, e)
    if expansion is not None:
        injected = _inflect_name(name, surface)
        return text[:s] + f"{injected} {expansion}" + text[e + clitic_len :], True
    return text[:s] + _inflect_name(name, surface) + text[e:], True


def _substitute_resolved(text: str, start, end, name: str) -> tuple[str, str]:
    """Substitute `name` into `text` over the resolved mention span, if allowed.

    Returns (text_to_score, outcome), where outcome is one of "substituted",
    "person", "not_mention", "no_span" or "bad_span".

    Every non-substituted outcome scores the unchanged text, leaving the model without
    an aspect anchor rather than with nonsense.
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
    if not entity_filter.is_substitutable_mention(surface):
        return text, "not_mention"
    new_text, ok = _substitute(text, start, end, name)
    if not ok:
        return text, "bad_span"
    return new_text, "substituted"


def build_pairs(sentences_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Turn a tagged sentence table into (text, aspect) scoring pairs.

    Adds absa_text and absa_aspect, rewriting nothing.

    An ABSA model attends to an aspect term that must occur in the text, so a sentence
    that only refers back to the company has the resolved name injected over the
    mention's own characters using its recorded span.

    Rules, in priority order:
      1. needs_score() False: empty pair, not scored
      2. target named explicitly: text unchanged, aspect is the display name
      3. target resolved by coref with a usable span: substitute the display name,
         inflected to the surface it replaces. Person spans, non-company surfaces and
         unusable spans fall back to unchanged text
      4. CEO and not target: aspect is the matched person alias, text unchanged

    One aspect per row: rule 4's guard makes the two tiers mutually exclusive.

    Returns a copy.
    """
    out = sentences_df.copy()
    if len(out) == 0:
        out["absa_text"] = pd.Series(dtype=object)
        out["absa_aspect"] = pd.Series(dtype=object)
        return out

    mask = sentiment.needs_score(out)
    target_name = _display_name(ticker)
    person_pattern = entity_filter._build_ticker_patterns(ticker)["person"]

    # Default: no pair. Rows the rules below never touch stay empty, which is
    # what "not scored" looks like on this table.
    texts: list[str] = [""] * len(out)
    aspects: list[str] = [""] * len(out)
    outcomes = {
        "substituted": 0,
        "person": 0,
        "not_mention": 0,
        "no_span": 0,
        "bad_span": 0,
    }
    # Positional iteration: `mask` is index-aligned to `out`, so take its values.
    mask_values = mask.to_numpy()
    text_values = out["text"].tolist()
    m_target = out["mentions_target"].fillna(False).astype(bool).tolist()
    m_ceo = out["mentions_ceo"].fillna(False).astype(bool).tolist()
    resolved = (
        out.get("resolved_by_coref", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    ).tolist()
    span_start = out.get("mention_char_start", pd.Series([None] * len(out))).tolist()
    span_end = out.get("mention_char_end", pd.Series([None] * len(out))).tolist()

    for i in range(len(out)):
        if not mask_values[i]:
            continue
        text = text_values[i] or ""

        if m_target[i]:
            name = target_name
            if name is None:
                continue
            if resolved[i]:
                new_text, outcome = _substitute_resolved(text, span_start[i], span_end[i], name)
                outcomes[outcome] += 1
                if outcome == "bad_span":
                    logger.debug(
                        f"build_pairs: unusable mention span "
                        f"({span_start[i]}, {span_end[i]}) on a {len(text)}-char sentence; "
                        "scoring the unchanged text"
                    )
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
        f"({len(out)} total); resolved rows: {outcomes['substituted']} substituted, "
        f"{outcomes['person']} left unchanged (person mention), "
        f"{outcomes['not_mention']} left unchanged (not a company mention), "
        f"{outcomes['no_span']} with no span recorded, "
        f"{outcomes['bad_span']} with an unusable span"
    )
    return out


def score_sentence_table(
    sentences_df: pd.DataFrame,
    ticker: str,
    cache_path=ABSA_CACHE_PATH,
) -> pd.DataFrame:
    """Build pairs for `sentences_df` and score them.

    Returns the table with absa_text, absa_aspect and the three absa_* score columns
    attached. Mirrors sentiment.score_sentence_table(): only needs_score() rows are
    scored, rows without a pair get NaN rather than 0 (0.0 is a real probability the
    model could return), and the cache is saved once at the end, merged.

    Returns a copy.
    """
    out = build_pairs(sentences_df, ticker)
    if len(out) == 0:
        for col in SCORE_COLUMNS:
            out[col] = pd.Series(dtype=float)
        return out

    cache_df = load_cache(cache_path)
    pairs = [(t, a) for t, a in zip(out["absa_text"].tolist(), out["absa_aspect"].tolist()) if a]
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


def score_headlines(
    headlines_df: pd.DataFrame,
    ticker: str,
    cache_path=ABSA_CACHE_PATH,
) -> pd.DataFrame:
    """Score each headline toward the target company, one row per input row.

    No substitution: a headline is standalone, so there is no antecedent to inject.
    Headlines not naming the target are still scored, an absent aspect returning
    near-neutral.

    Returns article_id plus SCORE_COLUMNS. The cache is merged, never replaced.
    """
    aspect = _display_name(ticker)
    if aspect is None:
        raise KeyError(f"ticker {ticker!r} has no 'names' entry in COMPANIES")

    out = headlines_df.copy()
    if len(out) == 0:
        for col in SCORE_COLUMNS:
            out[col] = pd.Series(dtype=float)
        return out[["article_id", *SCORE_COLUMNS]]

    texts = out["headline"].fillna("").astype(str).tolist()
    cache_df = load_cache(cache_path)
    scored = score_pairs([(t, aspect) for t in texts], cache_df=cache_df)
    save_cache(pd.concat([cache_df, scored], ignore_index=True), path=cache_path)

    # score_pairs() returns the raw model columns pos/neg/neu; the absa_
    # prefix is applied here, exactly as score_sentence_table() does, so the
    # two entry points hand back the same names.
    out["pair_hash"] = [pair_hash(t, aspect) for t in texts]
    merged = out.merge(scored.drop_duplicates("pair_hash"), on="pair_hash", how="left").drop(
        columns=["pair_hash"]
    )
    merged = merged.rename(columns=dict(zip(("pos", "neg", "neu"), SCORE_COLUMNS)))
    for col in SCORE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged[["article_id", *SCORE_COLUMNS]]
