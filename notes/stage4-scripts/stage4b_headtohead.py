"""Stage 4b: is Maverick BETTER than fastcoref, not merely different?

Part A -- head-to-head on the 270 hand-labelled rows (precision on fastcoref's picks).
Part B -- backend-swap over the same 200 articles: how many sentences would Maverick
          tag that fastcoref does not (the recall blind spot), and build the sample frame.

Reuses notes/stage4-scripts/ artifacts: the cached Maverick clusters and the
per-row agreement parquet. No new Maverick inference.
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\ML\stock-predictor")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from stock_predictor.text.entity_filter import (  # noqa: E402
    PERSON_ENT_LABEL,
    _build_ticker_patterns,
    _fix_missing_space,
    _get_nlp,
    map_coref_clusters,
    split_sentences,
    tag_sentences,
)


# wilson_ci -- exact definition from notebooks/text/2.7 cell 6
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def pct(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} = {100*k/n:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]"


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant pair (b, c)."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


ev = pd.read_parquet(ROOT / "data/eval/coref_eval_labelled.parquet")
mav = pd.read_parquet(ROOT / "data/interim/maverick_agreement.parquet")
ticker = ev["ticker"].iloc[0]
assert mav["mv_ok"].all()

# ---------------------------------------------------------------------------
# PART A -- head-to-head on the labelled rows
# ---------------------------------------------------------------------------
print("=" * 78)
print("PART A -- head-to-head on the 270 labelled rows (fastcoref's picks only)")
print("=" * 78)

df = mav.merge(
    ev[["row_id", "borderline"]].rename(columns={"borderline": "bl2"}), on="row_id", how="left"
)
df["fc_claims_target"] = True  # every eval row IS a fastcoref target pick, by construction
# Maverick's own resolution, mirroring stage4-maverick-findings.md 2.4:
#  span rows   -> would Maverick put a TARGET-keyed mention over the anaphor span
#                 (i.e. would it license the same rewrite)?
#  no-span rows-> would Maverick put a TARGET-keyed mention inside the sentence window?
df["mv_claims_target"] = np.where(
    df["has_span"], df["agree_span"].astype("boolean").fillna(False), df["agree_sentence"]
).astype(bool)
# secondary, looser reading for span rows: Maverick tags the SENTENCE (but maybe
# over a different anaphor). Reported alongside so the strict test is not the
# only definition on offer.
df["mv_claims_sentence"] = df["n_mv_target_in_sent"] > 0
df["human_target"] = df["verdict"] == "target"

rows = []
for label, sub in [
    ("ALL", df),
    ("span", df[df["has_span"]]),
    ("no-span", df[~df["has_span"]]),
    ("ALL excl borderline", df[~df["borderline"]]),
    ("span excl borderline", df[df["has_span"] & ~df["borderline"]]),
    ("no-span excl borderline", df[~df["has_span"] & ~df["borderline"]]),
]:
    n = len(sub)
    ht = sub["human_target"]
    mv = sub["mv_claims_target"]
    # fastcoref claims all n rows -> precision on this population
    fc_prec_k, fc_prec_n = int(ht.sum()), n
    mv_prec_k, mv_prec_n = int((mv & ht).sum()), int(mv.sum())
    # discordance
    both_right = int((mv & ht).sum())          # both claim target, human agrees
    fc_right_mv_wrong = int((~mv & ht).sum())  # mv drops a row the human called target
    fc_wrong_mv_right = int((~mv & ~ht).sum())  # mv drops a row the human called other
    both_wrong = int((mv & ~ht).sum())         # both claim target, human says other
    n_err = int((~ht).sum())
    print(f"\n--- {label}  (n={n}, errors={n_err}) ---")
    print(f"  fastcoref precision on this population : {pct(fc_prec_k, fc_prec_n)}")
    print(f"  Maverick precision on fastcoref's picks: {pct(mv_prec_k, mv_prec_n)}")
    print(f"  errors Maverick avoids (of {n_err})      : {pct(fc_wrong_mv_right, n_err)}")
    print(f"  correct rows Maverick would drop       : {pct(fc_right_mv_wrong, fc_prec_k)}")
    print(f"  discordant: fc-right/mv-wrong = {fc_right_mv_wrong}, "
          f"fc-wrong/mv-right = {fc_wrong_mv_right}, "
          f"McNemar exact p = {mcnemar_exact(fc_right_mv_wrong, fc_wrong_mv_right):.4f}")
    rows.append(
        dict(
            population=label, n=n, n_errors=n_err,
            fc_claims=fc_prec_n, fc_prec_k=fc_prec_k, fc_prec=fc_prec_k / fc_prec_n,
            fc_prec_lo=wilson_ci(fc_prec_k, fc_prec_n)[0], fc_prec_hi=wilson_ci(fc_prec_k, fc_prec_n)[1],
            mv_claims=mv_prec_n, mv_prec_k=mv_prec_k,
            mv_prec=(mv_prec_k / mv_prec_n) if mv_prec_n else float("nan"),
            mv_prec_lo=wilson_ci(mv_prec_k, mv_prec_n)[0], mv_prec_hi=wilson_ci(mv_prec_k, mv_prec_n)[1],
            both_right=both_right, both_wrong=both_wrong,
            fc_right_mv_wrong=fc_right_mv_wrong, fc_wrong_mv_right=fc_wrong_mv_right,
            mcnemar_p=mcnemar_exact(fc_right_mv_wrong, fc_wrong_mv_right),
        )
    )

# span rows under the looser sentence-level reading of "Maverick's resolution"
sub = df[df["has_span"]]
mv2 = sub["mv_claims_sentence"]
ht = sub["human_target"]
print("\n--- span rows, LOOSER definition (Maverick tags the sentence, span may differ) ---")
print(f"  Maverick precision: {pct(int((mv2 & ht).sum()), int(mv2.sum()))}")
print(f"  fc-right/mv-wrong = {int((~mv2 & ht).sum())}, fc-wrong/mv-right = {int((~mv2 & ~ht).sum())}")

summary = pd.DataFrame(rows)
summary.to_parquet(ROOT / "data/interim/stage4b_headtohead_summary.parquet", index=False)

df_out = df[[
    "row_id", "article_id", "sent_idx", "has_span", "borderline", "verdict",
    "human_target", "fc_claims_target", "mv_claims_target", "mv_claims_sentence",
    "mv_state", "agree_span", "agree_sentence", "n_fc_in_sent", "n_mv_target_in_sent",
    "n_mv_any_in_sent", "mv_chunked",
]].copy()
df_out["outcome"] = np.select(
    [
        df_out["mv_claims_target"] & df_out["human_target"],
        df_out["mv_claims_target"] & ~df_out["human_target"],
        ~df_out["mv_claims_target"] & df_out["human_target"],
        ~df_out["mv_claims_target"] & ~df_out["human_target"],
    ],
    ["both_right", "both_wrong", "fc_right_mv_wrong", "fc_wrong_mv_right"],
    default="?",
)
df_out.to_parquet(ROOT / "data/interim/stage4b_backend_headtohead.parquet", index=False)
print(f"\nwrote data/interim/stage4b_backend_headtohead.parquet ({len(df_out)} rows)")
print(df_out["outcome"].value_counts().to_string())

# ---------------------------------------------------------------------------
# PART B -- backend swap over the same 200 articles (the recall blind spot)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART B -- backend swap: what changes across ALL sentences of the 200 articles")
print("=" * 78)

# Stage 4's cached Maverick clusters. Produced by stage4_run.py, which writes them
# next to itself; if this script has been copied into the repo, point MV_CACHE at
# that scratchpad copy rather than re-running 200 articles of inference (~18 min).
MV_CACHE = SCRATCH / "maverick_clusters.pkl"
if not MV_CACHE.exists():
    raise SystemExit(
        f"{MV_CACHE} not found -- copy the pickle stage4_run.py produced next to this "
        "script, or re-run stage4_prep.py + stage4_run.py to regenerate it."
    )
with open(MV_CACHE, "rb") as f:
    mv_clusters = pickle.load(f)

aids = sorted(ev["article_id"].unique().tolist())
pa = pd.read_parquet(
    ROOT / "data/processed/processed_articles.parquet",
    columns=["article_id", "processed_body"],
)
pa = pa[pa["article_id"].isin(aids)].drop_duplicates("article_id").set_index("article_id")
cleaned = {a: _fix_missing_space(pa.loc[a, "processed_body"]) for a in aids if a in pa.index}
order = [a for a in aids if a in cleaned]
print(f"{len(order)} articles; maverick clusters cached for "
      f"{sum(1 for a in order if mv_clusters.get(a) is not None)}")

from stock_predictor.text import coref  # noqa: E402
from stock_predictor.text.sentiment import hash_text  # noqa: E402

cache = coref.load_cache()
lookup = dict(zip(cache["text_hash"], cache["clusters_json"]))
patterns = _build_ticker_patterns(ticker)["names"]

nlp = _get_nlp()
docs_spans = split_sentences([cleaned[a] for a in order], nlp=nlp, return_spans=True)

frames = []
for a, spans in zip(order, docs_spans):
    if not spans:
        continue
    txt = cleaned[a]
    doc = spans[0].doc
    person_spans = [(e.start_char, e.end_char) for e in doc.ents if e.label_ == PERSON_ENT_LABEL]
    h = hash_text(txt)
    fc_clusters = coref._deserialize_clusters(lookup[h]) if h in lookup else []
    mvc = mv_clusters.get(a) or []
    fc_m, _ = map_coref_clusters(fc_clusters, txt, ticker, patterns, person_spans=person_spans)
    mv_m, _ = map_coref_clusters(mvc, txt, ticker, patterns, person_spans=person_spans)
    t_fc = tag_sentences(a, spans, ticker, nlp=nlp, coref_mentions=fc_m)
    t_mv = tag_sentences(a, spans, ticker, nlp=nlp, coref_mentions=mv_m)
    m = t_fc[[
        "article_id", "sent_idx", "text", "mentions_target", "resolved_by_coref",
        "anaphor_char_start",
    ]].rename(columns={
        "mentions_target": "fc_target", "resolved_by_coref": "fc_coref",
        "anaphor_char_start": "fc_span",
    })
    m2 = t_mv[["sent_idx", "mentions_target", "resolved_by_coref", "anaphor_char_start"]].rename(
        columns={"mentions_target": "mv_target", "resolved_by_coref": "mv_coref",
                 "anaphor_char_start": "mv_span"}
    )
    frames.append(m.merge(m2, on="sent_idx"))

sw = pd.concat(frames, ignore_index=True)
sw["fc_has_span"] = sw["fc_span"].notna()
sw["mv_has_span"] = sw["mv_span"].notna()
print(f"\ntotal sentences across the {len(frames)} articles: {len(sw)}")

n_fc_t, n_mv_t = int(sw["fc_target"].sum()), int(sw["mv_target"].sum())
n_fc_c, n_mv_c = int(sw["fc_coref"].sum()), int(sw["mv_coref"].sum())
print(f"mentions_target      : fastcoref {n_fc_t}   maverick {n_mv_t}")
print(f"resolved_by_coref    : fastcoref {n_fc_c}   maverick {n_mv_c}")

both = int((sw["fc_coref"] & sw["mv_coref"]).sum())
fc_only = int((sw["fc_coref"] & ~sw["mv_coref"]).sum())
mv_only = int((~sw["fc_coref"] & sw["mv_coref"]).sum())
print("\ncoref-tagged sentences (the population the eval set samples):")
print(f"  both backends tag it        : {both}")
print(f"  fastcoref ONLY (mv drops)   : {fc_only}")
print(f"  MAVERICK ONLY (new rows)    : {mv_only}   <-- the recall blind spot")

# split the Maverick-only rows by whether they'd carry a rewrite
mv_only_mask = ~sw["fc_coref"] & sw["mv_coref"]
print(f"    of which span (rewritten) : {int((mv_only_mask & sw['mv_has_span']).sum())}")
print(f"    of which no-span          : {int((mv_only_mask & ~sw['mv_has_span']).sum())}")
fc_only_mask = sw["fc_coref"] & ~sw["mv_coref"]
print(f"  fastcoref-only, span        : {int((fc_only_mask & sw['fc_has_span']).sum())}")
print(f"  fastcoref-only, no-span     : {int((fc_only_mask & ~sw['fc_has_span']).sum())}")

# agreement on the SPAN itself among sentences both tag
bs = sw[sw["fc_coref"] & sw["mv_coref"] & sw["fc_has_span"] & sw["mv_has_span"]]
print(f"\n  both tag with a span: {len(bs)}, identical anaphor start: "
      f"{int((bs['fc_span'] == bs['mv_span']).sum())}")

# net effect on mentions_target (coref only fires where surface/CEO match did not)
print("\nnet target-set effect (mentions_target):")
print(f"  target under fastcoref only : {int((sw['fc_target'] & ~sw['mv_target']).sum())}")
print(f"  target under maverick only  : {int((~sw['fc_target'] & sw['mv_target']).sum())}")
print(f"  target under both           : {int((sw['fc_target'] & sw['mv_target']).sum())}")

sw.to_parquet(ROOT / "data/interim/stage4b_backend_swap_sentences.parquet", index=False)
print(f"\nwrote data/interim/stage4b_backend_swap_sentences.parquet ({len(sw)} rows)")

# ---- sample frame for the rows nobody has labelled: Maverick-only ----
frame = sw[mv_only_mask].copy()
frame["row_kind"] = np.where(frame["mv_has_span"], "span", "no-span")
frame = frame[[
    "article_id", "sent_idx", "text", "row_kind", "mv_span", "fc_target", "mv_target",
]]
frame.to_parquet(ROOT / "data/interim/stage4b_maverick_only_frame.parquet", index=False)
print(f"wrote data/interim/stage4b_maverick_only_frame.parquet ({len(frame)} rows) "
      "-- the unlabelled population; NOT labelled here")

# corpus-level projection inputs (measured on 200 articles; scaling is estimated)
print("\nper-article rates over the 200 eval articles (for projection):")
print(f"  fc coref-tagged / article : {n_fc_c / len(frames):.2f}")
print(f"  mv coref-tagged / article : {n_mv_c / len(frames):.2f}")
print(f"  mv-only / article         : {mv_only / len(frames):.2f}")
print(f"  fc-only / article         : {fc_only / len(frames):.2f}")
