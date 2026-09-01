"""The layer join: text features and market features into one model-ready table.

The two pipelines meet here and nowhere else. `text/` never imports `market/`
and neither imports this package; all three communicate through parquet files
keyed on `article_id`.

Modules:
    integrity.py    is `article_id` still one article, everywhere?
    leakage.py      notebook 2.0's momentum_1d regression test, per ticker
    merge.py        the join, and the accounting for what it dropped
    report.py       one report per ticker
    run_pipeline.py corpus in, merged.parquet out, per ticker plus pooled
"""
