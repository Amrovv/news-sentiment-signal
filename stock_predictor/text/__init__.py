"""Person A — Text & Deep Learning layer.

Delivers data/processed/articles.parquet (one row per article), joined by
Person B on `article_id`. This package never imports `market/`; the layers
meet only at the Parquet tables and shared constants in `config.py`.

Modules (promote from notebooks/a_*.ipynb when stable / reused):
    collect.py      Finnhub corpus pull          -> data/raw/raw_articles.parquet
    entity.py       spaCy sentence + alias filter
    analyze.py      analyze() FinBERT scoring     (reused: batch AND live demo)
    embeddings.py   768-dim vectors + novelty
    topics.py       BERTopic + topic-conditioned sentiment
    finetune.py     stage (c) fine-tune
    build_table.py  run pipeline over corpus      -> data/processed/articles.parquet
"""
