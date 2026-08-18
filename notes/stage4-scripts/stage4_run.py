"""Stage 4: run Maverick over the eval articles and emit per-row agreement flags.

Writes data/interim/maverick_agreement.parquet keyed (article_id, sent_idx).
"""
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\ML\stock-predictor")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(ROOT))

from mvload import load_maverick  # noqa: E402
from stock_predictor.text.entity_filter import (  # noqa: E402
    _build_ticker_patterns,
    map_coref_clusters,
)

with open(SCRATCH / "stage4_prep.pkl", "rb") as f:
    prep = pickle.load(f)

ev = pd.read_parquet(ROOT / "data/eval/coref_eval_labelled.parquet")
ticker = ev["ticker"].iloc[0]
patterns = _build_ticker_patterns(ticker)["names"]

MV_CACHE = SCRATCH / "maverick_clusters.pkl"
mv_clusters = {}
if MV_CACHE.exists():
    with open(MV_CACHE, "rb") as f:
        mv_clusters = pickle.load(f)

CHUNK_ABOVE_ = 1800
_size = {a: sum(len(s) for s in d["tok_sents"]) for a, d in prep.items()}
# the first partial run predated chunking; drop any cached long article so every
# article above the threshold is produced the same way
mv_clusters = {a: v for a, v in mv_clusters.items() if _size.get(a, 0) <= CHUNK_ABOVE_}

todo = [a for a in prep if a not in mv_clusters]
print(f"{len(todo)} articles to run through Maverick ({len(mv_clusters)} cached)")

# Maverick's mention scorer is O(n^2) in subtokens and a single forward pass
# over a 10k-token article blew past 9GB RSS and stalled on this 16GB box. Long
# articles are therefore split into consecutive, sentence-aligned blocks and the
# per-block clusters are unioned. Cross-block chains are lost, which can only
# ADD disagreement, so every chunked article is flagged (mv_chunked) and the
# headline numbers are reported with and without them.
CHUNK_ABOVE = 1800
CHUNK_SIZE = 1200

chunked = set()


def run_article(mv, d):
    """-> (clusters as absolute char spans, was_chunked)."""
    sizes = [len(s) for s in d["tok_sents"]]
    total = sum(sizes)
    blocks = []  # list of (sent_lo, sent_hi)
    if total <= CHUNK_ABOVE:
        blocks = [(0, len(sizes))]
    else:
        lo, acc = 0, 0
        for i, sz in enumerate(sizes):
            acc += sz
            if acc >= CHUNK_SIZE:
                blocks.append((lo, i + 1))
                lo, acc = i + 1, 0
        if lo < len(sizes):
            blocks.append((lo, len(sizes)))
    out = []
    for lo, hi in blocks:
        sents = d["tok_sents"][lo:hi]
        flat = [c for s in d["tok_chars"][lo:hi] for c in s]
        if not flat:
            continue
        res = mv.predict(sents)
        out.extend(
            [(flat[s][0], flat[e][1]) for s, e in cluster]
            for cluster in res["clusters_token_offsets"]
        )
    return out, len(blocks) > 1


if todo:
    mv = load_maverick()
    t0 = time.time()
    for n, a in enumerate(todo, 1):
        d = prep[a]
        try:
            cl, was_chunked = run_article(mv, d)
            if was_chunked:
                chunked.add(a)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {a}: {type(exc).__name__}: {exc}", flush=True)
            cl = None
        mv_clusters[a] = cl
        if n % 5 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el:.0f}s elapsed, {el / n:.2f}s/article", flush=True)
            with open(MV_CACHE, "wb") as f:
                pickle.dump(mv_clusters, f)
    with open(MV_CACHE, "wb") as f:
        pickle.dump(mv_clusters, f)

n_failed = sum(1 for v in mv_clusters.values() if v is None)
print(f"Maverick done. failures: {n_failed}")

# ---------------------------------------------------------------- agreement
def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def contained(m, off, length):
    return m[0] >= off and m[1] <= off + length


rows = []
for r in ev.itertuples():
    d = prep.get(r.article_id)
    mv_cl = mv_clusters.get(r.article_id)
    rec = {
        "row_id": r.row_id,
        "article_id": r.article_id,
        "sent_idx": r.sent_idx,
        "has_span": bool(r.has_span),
        "verdict": r.verdict,
        "borderline": bool(r.borderline),
        "mv_ok": mv_cl is not None and d is not None,
        "article_tokens": _size.get(r.article_id, 0),
        "mv_chunked": _size.get(r.article_id, 0) > CHUNK_ABOVE_,
    }
    if not rec["mv_ok"]:
        rows.append({**rec, "mv_state": "error"})
        continue

    txt = d["text"]
    off, sent_text = d["sents"][r.sent_idx]
    length = len(sent_text)

    mv_mentions, mv_stats = map_coref_clusters(
        mv_cl, txt, ticker, patterns, person_spans=d["person_spans"]
    )
    mv_target = [(s, e) for s, e, k in mv_mentions if k == "TARGET"]
    mv_all = [m for cluster in mv_cl for m in cluster]

    fc_target = [(s, e) for s, e, k in d["fc_mentions"] if k == "TARGET"]

    fc_in_sent = [m for m in fc_target if contained(m, off, length)]
    mv_in_sent = [m for m in mv_target if contained(m, off, length)]
    mv_any_in_sent = [m for m in mv_all if contained(m, off, length)]

    # fraction of fastcoref's in-sentence target mentions that Maverick also
    # places in a target-keyed chain
    if fc_in_sent:
        supported = sum(
            1 for f in fc_in_sent if any(overlaps(f, m) for m in mv_target)
        )
        support = supported / len(fc_in_sent)
    else:
        support = float("nan")

    rec.update(
        {
            "n_mv_clusters": len(mv_cl),
            "n_mv_target_mentions": len(mv_target),
            "n_fc_target_mentions": len(fc_target),
            "n_fc_in_sent": len(fc_in_sent),
            "n_mv_target_in_sent": len(mv_in_sent),
            "n_mv_any_in_sent": len(mv_any_in_sent),
            "mv_support_frac": support,
            # sentence-level criterion (used for no-span rows)
            "agree_sentence": len(mv_in_sent) > 0,
        }
    )

    if r.has_span and not pd.isna(r.anaphor_char_start):
        aspan = (off + int(r.anaphor_char_start), off + int(r.anaphor_char_end))
        rec["anaphor_abs_start"] = aspan[0]
        rec["anaphor_abs_end"] = aspan[1]
        rec["anaphor_surface"] = txt[aspan[0] : aspan[1]]
        rec["agree_span"] = any(overlaps(aspan, m) for m in mv_target)
        rec["mv_covers_anaphor"] = any(overlaps(aspan, m) for m in mv_all)
    else:
        rec["agree_span"] = None
        rec["mv_covers_anaphor"] = None

    # tri-state, on the criterion appropriate to the row type
    agree = rec["agree_span"] if r.has_span else rec["agree_sentence"]
    if agree:
        rec["mv_state"] = "support"
    else:
        touched = rec["mv_covers_anaphor"] if r.has_span else (len(mv_any_in_sent) > 0)
        rec["mv_state"] = "other_chain" if touched else "no_mention"
    rows.append(rec)

out = pd.DataFrame(rows)
dest = ROOT / "data/interim/maverick_agreement.parquet"
out.to_parquet(dest, index=False)
print(f"wrote {dest} ({len(out)} rows)")
print(out["mv_state"].value_counts())
