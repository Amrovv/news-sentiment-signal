"""Text layer: article bodies in, one feature row per article out.

Joined to the market layer on `article_id`; never imports `market/`. The two
layers meet only at the parquet tables and at `config.py`.

Modules in pipeline order:

    entity_filter.py  sentence split, entity tagging, coref mapping, boilerplate
                      flagging -> the 10-column SENTENCE_COLUMNS schema
    coref.py          fastcoref wrapper and cluster cache, backend for the above
    sentiment.py      FinBERT scoring, headline scoring, article aggregation
    absa.py           aspect-based sentiment (DeBERTa) with name substitution
    fusion.py         FinBERT + ABSA -> the shipped per-sentence score
    coref_judge.py    Qwen2.5-7B referent verification (llama.cpp)
    coref_eval.py     measures a judge against hand labels; gates coref_judge
    run_pipeline.py   the driver: `python -m news_sentiment.text.run_pipeline`

Every model-backed module degrades gracefully: is_available() returns False and
the stage's columns are absent, rather than raising.
"""
