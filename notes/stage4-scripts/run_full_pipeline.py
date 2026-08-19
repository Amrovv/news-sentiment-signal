"""Full text pipeline from processed_articles.parquet, with every Stage 1-4 change applied.

WHAT THIS IS. Notebook 2.3's pipeline, re-run end to end from the article corpus
rather than from the existing sentence table, plus the three things this branch
added: the referent-verification judge (Stage 1), the provenance split (Stage 2),
and the fusion aggregates. It exists so the branch's claims can be checked
against a corpus produced by the current code in one pass, instead of against a
sentence table produced weeks ago and patched since.

    processed_articles.parquet (2,124)
      -> entity_filter.process_articles   split + tag + coref + boilerplate
      -> sentiment.score_sentence_table   FinBERT   (cached)
      -> sentiment.score_headlines        FinBERT   (cached)
      -> absa.score_sentence_table        DeBERTa   (cached)
      -> coref_judge                      Qwen2.5   (cached)  <- the slow part
      -> sentiment.aggregate_article_features
      -> fusion.aggregate_fusion_features + aggregate_provenance_features

WRITES TO data/interim/full_run/, NEVER over data/sentences.parquet. The
committed corpus stays untouched until the two have been compared deliberately;
overwriting it as a side effect of a verification run would destroy the very
thing being verified against.

RESUMABLE. Each phase skips if its output exists, and the judge flushes verdicts
to the shared on-disk cache every CHUNK rows. The judge cache is keyed
(article_id, sent_idx, target, model_id, prompt_version) with merge-never-replace
semantics, so an interrupted run resumes rather than restarting -- at ~4.9 s/row
over ~3.5k coref sentences that matters.
"""
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_predictor.config import PRIMARY_TICKER, PROCESSED_DATA_DIR
from stock_predictor.text import absa, entity_filter, fusion, sentiment
from stock_predictor.text.coref_eval import (
    CACHE_COLUMNS,
    build_context,
    load_judge_cache,
    save_judge_cache,
)
from stock_predictor.text.coref_judge import PROMPT_VERSION, make_judge

OUT = Path("data/interim/full_run")
OUT.mkdir(parents=True, exist_ok=True)
CHUNK = 100
MODEL_ID = "qwen2.5-7b-instruct-q4km"

TAGGED = OUT / "sentences_tagged.parquet"
SCORED = OUT / "sentences_scored.parquet"
JUDGED = OUT / "sentences_judged.parquet"

articles = pd.read_parquet(PROCESSED_DATA_DIR / "processed_articles.parquet")
logger.info(f"corpus: {len(articles)} articles")

# --- Phase A: split, tag, coref, boilerplate -------------------------------
if TAGGED.exists():
    sentences_df = pd.read_parquet(TAGGED)
    logger.info(f"[A] reusing {TAGGED} ({len(sentences_df)} sentences)")
else:
    t0 = time.time()
    sentences_df = entity_filter.process_articles(
        articles, ticker=PRIMARY_TICKER, text_col="processed_body"
    )
    logger.info(
        f"[A] tagging: {time.time() - t0:.0f}s -- {len(sentences_df)} sentences "
        f"from {articles['article_id'].nunique()} articles"
    )
    sentences_df.to_parquet(TAGGED, index=False)

# --- Phase B: FinBERT + ABSA ------------------------------------------------
if SCORED.exists():
    scored = pd.read_parquet(SCORED)
    logger.info(f"[B] reusing {SCORED}")
else:
    t0 = time.time()
    scored = sentiment.score_sentence_table(sentences_df)
    logger.info(f"[B] FinBERT sentences: {time.time() - t0:.0f}s")
    t0 = time.time()
    scored = absa.score_sentence_table(scored, ticker=PRIMARY_TICKER)
    logger.info(f"[B] ABSA: {time.time() - t0:.0f}s")
    scored.to_parquet(SCORED, index=False)

headline_scores = sentiment.score_headlines(articles[["article_id", "headline"]])

# --- Phase C: the judge over every coref sentence ---------------------------
scored["provenance_channel"] = fusion.provenance_channel(scored)
judge_pop = scored[
    scored["mentions_target"].fillna(False)
    & ~scored["is_boilerplate"].fillna(False)
    & scored["resolved_by_coref"].fillna(False)
]
logger.info(
    f"[C] judge population: {len(judge_pop)} coref sentences over "
    f"{judge_pop['article_id'].nunique()} articles "
    f"({judge_pop['provenance_channel'].value_counts().to_dict()})"
)

cache = load_judge_cache()
mine = cache[(cache["model_id"] == MODEL_ID) & (cache["prompt_version"] == PROMPT_VERSION)]
answered = set(zip(mine["article_id"], mine["sent_idx"]))
todo = judge_pop[~pd.Series(
    list(zip(judge_pop["article_id"], judge_pop["sent_idx"])), index=judge_pop.index
).isin(answered)]
logger.info(f"[C] {len(answered)} cached under {PROMPT_VERSION}; {len(todo)} to judge")

if len(todo):
    head_col = "headline" if "headline" in articles.columns else "title"
    headlines = articles.set_index("article_id")[head_col].to_dict()
    by_article = {aid: g.sort_values("sent_idx") for aid, g in scored.groupby("article_id")}

    judge = make_judge()
    fresh, start = [], time.time()
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        ctx = build_context(
            article_id=row.article_id,
            sent_idx=row.sent_idx,
            sentences=by_article[row.article_id],
            headlines=headlines,
            target=PRIMARY_TICKER,
            anaphor_char_start=getattr(row, "anaphor_char_start", None),
            anaphor_char_end=getattr(row, "anaphor_char_end", None),
        )
        fresh.append({
            "article_id": int(row.article_id), "sent_idx": int(row.sent_idx),
            "target": PRIMARY_TICKER, "model_id": MODEL_ID,
            "prompt_version": PROMPT_VERSION, "answer": judge(ctx),
        })
        if i % CHUNK == 0 or i == len(todo):
            # Merge, never replace -- `fresh` is only this run's new verdicts.
            save_judge_cache(pd.concat(
                [load_judge_cache(), pd.DataFrame(fresh, columns=CACHE_COLUMNS)],
                ignore_index=True,
            ))
            fresh = []
            rate = (time.time() - start) / i
            logger.info(f"[C] {i}/{len(todo)} ({i/len(todo):.1%}) "
                        f"{rate:.2f}s/row ~{(len(todo)-i)*rate/3600:.1f}h left")

# --- Phase D: attach verdicts and aggregate ---------------------------------
cache = load_judge_cache()
mine = cache[(cache["model_id"] == MODEL_ID) & (cache["prompt_version"] == PROMPT_VERSION)]
scored = scored.merge(
    mine[["article_id", "sent_idx", "answer"]].rename(columns={"answer": "judge_answer"}),
    on=["article_id", "sent_idx"], how="left",
)
# A sentence survives the gate if it is not coref-resolved (the judge has no
# opinion on surface matches) or the judge said yes. Fail closed: anything else
# -- no, unsure, or a coref row the judge never saw -- is dropped.
is_coref = scored["resolved_by_coref"].fillna(False)
scored["judge_accepted"] = (~is_coref) | (scored["judge_answer"] == "yes")
scored.to_parquet(JUDGED, index=False)
logger.info(f"[D] wrote {JUDGED}")

gated = scored[scored["judge_accepted"]]
for tag, frame in [("ungated", scored), ("judge_gated", gated)]:
    # aggregate_article_features() ALREADY calls fusion.aggregate_fusion_features()
    # internally, so merging it again here produced duplicate fus_* columns with
    # _x/_y suffixes (values identical, verified). Only the provenance split is
    # genuinely additional.
    agg = sentiment.aggregate_article_features(frame, headline_scores=headline_scores)
    prov = fusion.aggregate_provenance_features(frame)
    out = (articles[["article_id", "timestamp_utc", "headline", "source"]]
           .assign(ticker=PRIMARY_TICKER)
           .merge(agg, on="article_id", how="left")
           .merge(prov, on="article_id", how="left"))
    out.to_parquet(OUT / f"articles_{tag}.parquet", index=False)
    logger.info(f"[D] wrote articles_{tag}.parquet {out.shape}")

logger.info("full pipeline run complete")
