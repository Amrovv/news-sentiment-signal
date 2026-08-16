"""Neural coreference resolution over article bodies (entity_filter backend).

WHAT THIS BUYS US OVER THE HEURISTIC. entity_filter.resolve_anaphora() fills
un-named sentences ("The company also raised prices.", "It expects...") by
handing them the most recently named company, with a decay window
(ANAPHORA_MAX_GAP) so a stale antecedent does not drift forever. That rule is a
positional proxy for a discourse fact, it was hand-scored at ~60% accuracy, and
it supplies roughly a third of all target sentences. Its two known failure modes
are exactly the ones recency cannot express:

  * multi-company roundups, where the nearest named company is an incidental
    mention rather than the topic ("...led by Peter Rawlinson, Tesla's former
    chief vehicle engineer" hijacking the following Lucid paragraph);
  * attribution inside subordinate clauses, where the company that is
    grammatically near is not the company the pronoun refers back to.

A coreference model reads the mention CHAIN instead of guessing: "The company"
is linked to the antecedent the model believes it refers to, across arbitrary
distance and across intervening mentions of other firms. We do not assume this
is better — resolved_by_coref is kept as a separate column from
resolved_by_anaphora precisely so the two mechanisms can be measured against
each other on the same corpus.

SPANS ARE CHARACTER OFFSETS INTO THE ORIGINAL TEXT that was passed in — not
token indices, not offsets into some normalised copy. That is deliberate and
load-bearing twice over:
  1. entity_filter maps mention spans onto sentence boundaries, which it also
     addresses by absolute character offset into the same string (a spaCy Doc
     built from it). Caller and callee must therefore be handed the SAME string
     — in the pipeline that is the _fix_missing_space()-cleaned body, never the
     raw one.
  2. the spans are reusable downstream: the ABSA work that follows needs to
     substitute the resolved company name into the sentence before target-aware
     scoring, which needs to know exactly which characters to replace.

GRACEFUL DEGRADATION. Every entry point here is best-effort. If fastcoref is not
installed, or the model cannot be loaded, or inference raises, is_available()
returns False (after one warning) and resolve_documents() returns empty cluster
lists. entity_filter then falls back to the heuristic for every sentence. The
pipeline must never hard-fail because coreference is unavailable.

ON-DISK CACHE. Coref inference is deterministic in the document text and is the
single most expensive stage in the text pipeline (~600s over the corpus, more
than FinBERT), so resolve_documents() caches its output exactly the way
sentiment.py caches FinBERT scores: a parquet keyed by
sentiment.hash_text(text) — the SAME hashing implementation, deliberately not a
second copy — with the clusters serialised into a JSON string column. Clusters
are list[list[tuple[int, int]]]; JSON avoids parquet's nested-dtype pain and
keeps the file trivially inspectable. The key is the string coref actually
sees (the _fix_missing_space()-cleaned body that process_articles builds), not
the raw article body, or a hit would return offsets into a different string.
Saves MERGE with the cache loaded at the start of the call and never replace it
— sentiment.py had a truncation bug from replacing, and the same shape of bug
here would silently throw away a full corpus run.

API:
    is_available()       -> bool, whether the backend imported and loaded
    load_cache()         -> the on-disk cluster cache (empty frame on any error)
    save_cache()         -> persist, deduped on text_hash (keep last)
    resolve_documents()  -> per document, a list of clusters of (start, end)
"""

import json
import logging

import pandas as pd
from loguru import logger

from stock_predictor.config import COREF_BATCH_SIZE, COREF_CACHE_PATH, COREF_MODEL
from stock_predictor.text.sentiment import hash_text

# Cached model handle, mirroring entity_filter._NLP: loading F-Coref costs a few
# seconds and allocates ~90M parameters, so it is loaded once per process.
# _LOAD_FAILED latches so a missing/broken backend is reported exactly once
# rather than re-attempting (and re-warning) on every batch.
_MODEL = None
_LOAD_FAILED = False

# Cluster spans are (char_start, char_end) into the document text.
Span = tuple[int, int]
Cluster = list[Span]

CACHE_COLUMNS = ["text_hash", "clusters_json"]


def _load_model():
    """Load (once, then memoise) the fastcoref model. Returns None on failure.

    Two compatibility notes, both the reason this is wrapped rather than a bare
    import:

      * fastcoref 2.1.x predates transformers 5.x. Its FCorefModel does not
        declare `all_tied_weights_keys`, which transformers' newer
        from_pretrained() finalisation reads unconditionally, so loading raises
        AttributeError. The class-level shim below supplies the empty mapping
        the base class would have (F-Coref ties no weights). It is applied
        defensively — if a future fastcoref/transformers pairing already
        provides the attribute, setdefault-style behaviour leaves it alone.
      * fastcoref logs at INFO through the stdlib logging module (per-request
        HTTP lines, tokenisation notices) which would swamp this project's
        loguru output, so its logger is quietened here.
    """
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    try:
        logging.getLogger("fastcoref").setLevel(logging.WARNING)
        from fastcoref import FCoref
        from fastcoref.modeling import FCorefModel

        if not hasattr(FCorefModel, "all_tied_weights_keys"):
            FCorefModel.all_tied_weights_keys = {}

        logger.info(f"Loading coreference model '{COREF_MODEL}' (CPU)")
        _MODEL = FCoref(model_name_or_path=COREF_MODEL, device="cpu", enable_progress_bar=False)
        logger.info("Coreference model loaded")
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
        _LOAD_FAILED = True
        logger.warning(
            f"Coreference backend unavailable ({type(exc).__name__}: {exc}); "
            "falling back to the anaphora heuristic"
        )
        _MODEL = None
    return _MODEL


def is_available() -> bool:
    """True when the coref backend imported AND the model loaded successfully.

    Callers use this as the enable-gate; a False here must always mean "use the
    heuristic", never "raise".
    """
    return _load_model() is not None


def _serialize_clusters(clusters: list[Cluster]) -> str:
    """Clusters -> JSON string, the on-disk form of the cache's value column."""
    return json.dumps([[[int(s), int(e)] for s, e in cluster] for cluster in clusters])


def _deserialize_clusters(payload) -> list[Cluster]:
    """JSON string -> clusters, tolerating a garbage value by returning [].

    A single unparseable row must not take down a whole run: the caller treats
    an empty cluster list as "coref produced nothing for this document", which
    is exactly the graceful-degradation behaviour the rest of the module has.
    """
    try:
        raw = json.loads(payload)
        return [[(int(s), int(e)) for s, e in cluster] for cluster in raw]
    except Exception:  # noqa: BLE001 - a corrupt row degrades, never raises
        return []


def load_cache(path=COREF_CACHE_PATH) -> pd.DataFrame:
    """Load the on-disk coref cluster cache.

    Returns an empty frame with CACHE_COLUMNS when the file does not exist.
    NEVER raises: a corrupt, truncated or schema-drifted file logs one warning
    and is treated as empty, so the run just pays for a cold pass instead of
    dying on a cache it could have ignored. (This is deliberately laxer than
    sentiment.load_cache(), which raises on a missing column — coref's cache is
    a pure speed optimisation with no effect on output values, so there is
    nothing to protect by failing loudly.)
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
            f"Coreference cache at {path} is unreadable "
            f"({type(exc).__name__}: {exc}); treating it as empty"
        )
        return pd.DataFrame(columns=CACHE_COLUMNS)


def save_cache(cache_df: pd.DataFrame, path=COREF_CACHE_PATH) -> None:
    """Persist the cache, deduped on text_hash (keep last).

    Callers must pass the MERGE of the cache they loaded and the rows they
    produced, never the new rows alone — see the module docstring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped = cache_df.drop_duplicates(subset="text_hash", keep="last")[CACHE_COLUMNS]
    deduped.to_parquet(path, index=False)
    logger.info(f"Saved coreference cache with {len(deduped)} entries to {path}")


def resolve_documents(
    texts: list[str],
    batch_size: int = COREF_BATCH_SIZE,
    cache_path=COREF_CACHE_PATH,
    use_cache: bool = True,
) -> list[list[Cluster]]:
    """Resolve coreference over a batch of documents, with an on-disk cache.

    Returns one list of clusters per input document, in input order; each
    cluster is a list of (char_start, char_end) mention spans into THAT
    document's text (see the module docstring on why char offsets).

    CACHING. Inputs are hashed with sentiment.hash_text() and split into hits
    (served straight from `cache_path`) and misses (sent to the model). Only the
    misses are run, and the new rows are MERGED with the cache that was loaded
    at the start of the call before a single save at the end — never written on
    their own, which would truncate the file to just this call's documents.
    Duplicate texts within one call collapse to one inference, like
    sentiment.score_sentences() does. Pass use_cache=False to bypass the file
    entirely (tests, and any caller that wants a guaranteed cold run).

    If every document is a cache hit the model is never loaded at all, so a warm
    run costs a parquet read and some JSON parsing.

    Batching goes through fastcoref's own predict() batcher, which packs
    documents up to `batch_size` SUBWORD TOKENS per forward pass — a per-document
    Python loop would forfeit that packing and run several times slower on a
    corpus of short news articles.

    Never raises: on any backend failure every unresolved document gets an empty
    cluster list, which entity_filter reads as "coref produced nothing here",
    and nothing is written to the cache.
    """
    if not texts:
        return []

    out: list[list[Cluster]] = [[] for _ in texts]

    # fastcoref chokes on empty/non-string inputs; keep the positional mapping
    # so the returned list still lines up 1:1 with `texts`.
    keep_idx = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    if not keep_idx:
        return out

    cache_df = load_cache(cache_path) if use_cache else pd.DataFrame(columns=CACHE_COLUMNS)
    cache_lookup: dict[str, str] = {}
    if len(cache_df):
        cache_lookup = {
            row.text_hash: row.clusters_json for row in cache_df.itertuples(index=False)
        }

    hashes = {i: hash_text(texts[i]) for i in keep_idx}
    # hash -> one representative text; duplicate documents are inferred once.
    to_run: dict[str, str] = {}
    n_hits = 0
    for i in keep_idx:
        h = hashes[i]
        if h in cache_lookup:
            out[i] = _deserialize_clusters(cache_lookup[h])
            n_hits += 1
        else:
            to_run.setdefault(h, texts[i])

    logger.info(
        f"resolve_documents: {len(keep_idx)} documents "
        f"({n_hits} cache hits, {len(to_run)} unique texts to resolve)"
    )

    if not to_run:
        return out

    model = _load_model()
    if model is None:
        return out

    run_hashes = list(to_run)
    payload = [to_run[h] for h in run_hashes]

    logger.info(
        f"Resolving coreference over {len(payload)} documents "
        f"(max {batch_size} tokens per batch)"
    )
    try:
        preds = model.predict(texts=payload, max_tokens_in_batch=batch_size)
    except Exception as exc:  # noqa: BLE001 - inference failure must degrade
        logger.warning(
            f"Coreference inference failed ({type(exc).__name__}: {exc}); "
            "falling back to the anaphora heuristic"
        )
        return out

    resolved: dict[str, list[Cluster]] = {}
    n_clusters = 0
    for h, pred in zip(run_hashes, preds):
        clusters = [
            [(int(s), int(e)) for s, e in cluster]
            for cluster in pred.get_clusters(as_strings=False)
        ]
        resolved[h] = clusters
        n_clusters += len(clusters)

    for i in keep_idx:
        h = hashes[i]
        if h in resolved:
            out[i] = resolved[h]

    logger.info(f"Coreference produced {n_clusters} clusters across {len(payload)} documents")

    if use_cache and resolved:
        new_rows = pd.DataFrame(
            [
                {"text_hash": h, "clusters_json": _serialize_clusters(c)}
                for h, c in resolved.items()
            ],
            columns=CACHE_COLUMNS,
        )
        # Merge, never replace: `resolved` covers only this call's misses, so
        # saving it alone would drop every previously cached document.
        save_cache(pd.concat([cache_df, new_rows], ignore_index=True), path=cache_path)

    return out
