"""Full text pipeline: the article table in, one model-facing feature row out.

    A  entity_filter.process_articles   split, tag, coref, boilerplate
    B  sentiment.score_sentence_table   FinBERT      (cached)
       absa.score_sentence_table        DeBERTa      (cached)
    C  coref_judge.judge_corpus         Qwen2.5      (cached, ~5h, the slow part)
    D  sentiment.aggregate_article_features + fusion.aggregate_provenance_features
    E  sentiment.build_model_features   -> the deliverable

Phases A to D write intermediates to data/interim/full_run/. Phase E writes
article_features.parquet and its generated data dictionary to
data/processed/pipeline_run/; that pair is the only output meant to leave the
text layer.

Resumable. A and B skip if their output exists, and the judge flushes verdicts
every JUDGE_CHUNK rows, so an interrupt costs at most that many.

Run with `python -m stock_predictor.text.run_pipeline`. Nothing runs on import.
"""

import time

from loguru import logger
import pandas as pd

from stock_predictor.config import (
    INTERIM_DATA_DIR,
    PRIMARY_TICKER,
    PROCESSED_ARTICLES_PATH,
    PROCESSED_DATA_DIR,
)
from stock_predictor.text import absa, coref_judge, entity_filter, fusion, sentiment

OUT = INTERIM_DATA_DIR / "full_run"
FINAL_DIR = PROCESSED_DATA_DIR / "pipeline_run"

TAGGED = OUT / "sentences_tagged.parquet"
SCORED = OUT / "sentences_scored.parquet"
JUDGED = OUT / "sentences_judged.parquet"
FINAL = FINAL_DIR / "article_features.parquet"
FINAL_DOC = FINAL_DIR / "article_features.md"

# Identity columns carried onto the final table. timestamp_utc is what the market
# layer joins on; source is a categorical the model may use.
IDENTITY_COLUMNS = ["article_id", "ticker", "timestamp_utc", "source"]


def write_feature_dictionary(final: pd.DataFrame, path=FINAL_DOC) -> None:
    """Emit the data dictionary that ships beside article_features.parquet.

    Generated from the frame rather than maintained by hand, so coverage and
    column list cannot drift from the file they describe.
    """
    identity = [c for c in IDENTITY_COLUMNS if c in final.columns]
    lines = [
        "# `article_features.parquet`",
        "",
        (
            "One row per article, judge-gated. The final output of the text pipeline and the "
            "table a price model consumes."
        ),
        "",
        (
            f"**Rows:** {len(final):,} | **Columns:** {len(final.columns)} | "
            f"**Span:** {final['timestamp_utc'].min():%Y-%m-%d} to {final['timestamp_utc'].max():%Y-%m-%d}"
        ),
        "",
        (
            "Produced by `stock_predictor.text.run_pipeline`, Phase E. Every score is a fusion of "
            "FinBERT and DeBERTa ABSA at `CONF_FLOOR`; no raw probability triples and no provenance "
            "columns are carried. Sentences the referent judge rejected contribute to nothing here."
        ),
        "",
        "## Identity",
        "",
        "| column | type | description |",
        "|---|---|---|",
    ]
    idesc = {
        "article_id": "Publisher-assigned identifier, unique per row.",
        "ticker": "The target company the features were computed for.",
        "timestamp_utc": "Publication time. The join key for the market layer.",
        "source": "Publisher name.",
    }
    for c in identity:
        lines.append(f"| `{c}` | {final[c].dtype} | {idesc.get(c, '')} |")

    lines += [
        "",
        "## Features",
        "",
        (
            "`coverage` is the share of rows that are non-null. NaN means the population was empty, "
            "never that the score was zero."
        ),
        "",
        "| column | type | coverage | mean | std | description |",
        "|---|---|---:|---:|---:|---|",
    ]
    for c in sentiment.MODEL_FEATURE_COLUMNS:
        col = final[c]
        mean = f"{col.mean():.3f}" if col.dtype.kind in "fi" else ""
        std = f"{col.std():.3f}" if col.dtype.kind in "fi" else ""
        desc = sentiment.FEATURE_DESCRIPTIONS.get(c, "")
        lines.append(
            f"| `{c}` | {col.dtype} | {col.notna().mean():.1%} | {mean} | {std} | {desc} |"
        )

    lines += [
        "",
        "## Reading the columns",
        "",
        (
            "- **NaN over zero.** 0.0 is a real fused score, so an empty population is NaN rather "
            "than 0. The exception is `fus_conf_graft_floor_lead`, which is 0.0 when the opening "
            "window holds no target sentence: an article that does not mention the target early "
            "delivers no early sentiment, and that is a measurement."
        ),
        (
            "- **Headline-only articles** keep their row with body scores filled to 0.0. They are "
            "identifiable by `n_entity_sents == 0`, and no other row has that, so the fill is "
            "reversible."
        ),
        "- **Sign** follows `pos - neg`: positive is favourable to the target.",
        (
            "- **Relevance.** Articles with no target sentence in the body and no target named in the "
            "headline are dropped, which is why the row count is below the corpus size."
        ),
        "",
        (
            "Every design decision behind these columns is argued in `notebooks/text/2.2` and "
            "`notebooks/text/2.3`."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[E] wrote {path}")


def main() -> None:
    """Run every phase in order, reusing any phase output already on disk."""
    OUT.mkdir(parents=True, exist_ok=True)

    articles = pd.read_parquet(PROCESSED_ARTICLES_PATH)
    logger.info(f"corpus: {len(articles)} articles")

    # --- Phase A: split, tag, coref, boilerplate ---------------------------
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

    # --- Phase B: FinBERT + ABSA -------------------------------------------
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

    # --- Phase C: the judge over every coref sentence -----------------------
    scored["provenance_channel"] = fusion.provenance_channel(scored)
    scored = coref_judge.judge_corpus(scored, articles, target=PRIMARY_TICKER)

    # --- Phase D: aggregate -------------------------------------------------
    scored.to_parquet(JUDGED, index=False)
    logger.info(f"[D] wrote {JUDGED}")

    gated = scored[scored["judge_accepted"]]
    for tag, frame in [("ungated", scored), ("judge_gated", gated)]:
        # aggregate_article_features() already merges the fus_* columns; only the
        # provenance split is additional. Merging both would duplicate them.
        agg = sentiment.aggregate_article_features(frame, headline_scores=headline_scores)
        prov = fusion.aggregate_provenance_features(frame)
        out = (
            articles[["article_id", "timestamp_utc", "headline", "source"]]
            .assign(ticker=PRIMARY_TICKER)
            .merge(agg, on="article_id", how="left")
            .merge(prov, on="article_id", how="left")
        )
        out.to_parquet(OUT / f"articles_{tag}.parquet", index=False)
        logger.info(f"[D] wrote articles_{tag}.parquet {out.shape}")

    # --- Phase E: the model-facing table ------------------------------------
    # One row per article, judge-gated, with no raw probability triples and no
    # provenance columns: every score is a fusion of both scorers. This is the
    # file a downstream model reads.
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    headline_absa = absa.score_headlines(
        articles[["article_id", "headline"]], ticker=PRIMARY_TICKER
    )
    features = sentiment.build_model_features(
        gated,
        headline_finbert=headline_scores,
        headline_absa=headline_absa,
        headlines_df=articles[["article_id", "headline"]],
        ticker=PRIMARY_TICKER,
    )
    final = (
        articles[["article_id", "timestamp_utc", "source"]]
        .assign(ticker=PRIMARY_TICKER)
        .merge(features, on="article_id", how="inner")[
            IDENTITY_COLUMNS + sentiment.MODEL_FEATURE_COLUMNS
        ]
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    final.to_parquet(FINAL, index=False)
    logger.info(f"[E] wrote {FINAL} {final.shape}")
    write_feature_dictionary(final)

    logger.info("full pipeline run complete")


if __name__ == "__main__":
    main()
