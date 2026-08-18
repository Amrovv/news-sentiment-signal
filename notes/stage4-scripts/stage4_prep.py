"""Stage 4 prep: rebuild the exact pipeline coordinate system for the eval articles.

Produces, per eval article:
  - cleaned text (_fix_missing_space(processed_body)) -- the string every char
    offset in the pipeline addresses
  - spaCy sentence spans (same parse the pipeline uses), so sent_idx maps to a
    (offset, length) window
  - fastcoref clusters, read straight from the pipeline's own on-disk cache
    (data/interim/coref_cache.parquet) keyed by hash_text(cleaned) -- these are
    literally the clusters the shipped sentences.parquet was built from
  - PERSON entity spans from that parse

Writes a pickle to the scratchpad for the Maverick pass to consume.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\ML\stock-predictor")
sys.path.insert(0, str(ROOT))
SCRATCH = Path(__file__).parent

from stock_predictor.text import coref  # noqa: E402
from stock_predictor.text.entity_filter import (  # noqa: E402
    PERSON_ENT_LABEL,
    _build_ticker_patterns,
    _fix_missing_space,
    _get_nlp,
    map_coref_clusters,
    split_sentences,
)

ev = pd.read_parquet(ROOT / "data/eval/coref_eval_labelled.parquet")
aids = sorted(ev["article_id"].unique().tolist())
print(f"{len(ev)} eval rows over {len(aids)} articles")

pa = pd.read_parquet(
    ROOT / "data/processed/processed_articles.parquet",
    columns=["article_id", "processed_body"],
)
pa = pa[pa["article_id"].isin(aids)].drop_duplicates("article_id").set_index("article_id")
print(f"matched {len(pa)} bodies")

cleaned = {a: _fix_missing_space(pa.loc[a, "processed_body"]) for a in aids if a in pa.index}

nlp = _get_nlp()
order = [a for a in aids if a in cleaned]
docs_spans = split_sentences([cleaned[a] for a in order], nlp=nlp, return_spans=True)

# fastcoref clusters straight from the pipeline cache
cache = coref.load_cache()
lookup = dict(zip(cache["text_hash"], cache["clusters_json"]))
from stock_predictor.text.sentiment import hash_text  # noqa: E402

ticker = ev["ticker"].iloc[0]
patterns = _build_ticker_patterns(ticker)["names"]

out = {}
n_cache_miss = 0
for a, spans in zip(order, docs_spans):
    txt = cleaned[a]
    h = hash_text(txt)
    if h not in lookup:
        n_cache_miss += 1
        clusters = []
    else:
        clusters = coref._deserialize_clusters(lookup[h])
    doc = spans[0].doc if spans else None
    person_spans = (
        [(e.start_char, e.end_char) for e in doc.ents if e.label_ == PERSON_ENT_LABEL]
        if doc is not None
        else []
    )
    sents = []
    for i, sp in enumerate(spans):
        raw = sp.text
        lead = len(raw) - len(raw.lstrip())
        sents.append((sp.start_char + lead, raw.strip()))
    fc_mentions, _ = map_coref_clusters(clusters, txt, ticker, patterns, person_spans=person_spans)
    # token table for Maverick: sentence-tokenized, exact char spans preserved
    tok_sents, tok_chars = [], []
    for sp in spans:
        ts = [t for t in sp if not t.is_space]
        if not ts:
            continue
        tok_sents.append([t.text for t in ts])
        tok_chars.append([(t.idx, t.idx + len(t.text)) for t in ts])
    out[a] = {
        "text": txt,
        "sents": sents,
        "person_spans": person_spans,
        "fc_clusters": clusters,
        "fc_mentions": fc_mentions,
        "tok_sents": tok_sents,
        "tok_chars": tok_chars,
    }

print(f"cache misses: {n_cache_miss} / {len(order)}")

# --- verify sent_idx alignment against the eval rows ---
bad = 0
for r in ev.itertuples():
    d = out.get(r.article_id)
    if d is None or r.sent_idx >= len(d["sents"]):
        bad += 1
        continue
    if d["sents"][r.sent_idx][1] != r.text:
        bad += 1
        print("MISMATCH", r.row_id, r.article_id, r.sent_idx)
        print("  eval:", repr(r.text[:90]))
        print("  doc :", repr(d["sents"][r.sent_idx][1][:90]))
print(f"sentence alignment mismatches: {bad} / {len(ev)}")

with open(SCRATCH / "stage4_prep.pkl", "wb") as f:
    pickle.dump(out, f)
print("wrote stage4_prep.pkl")
