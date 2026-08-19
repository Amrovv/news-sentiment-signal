"""Person A — Text & Deep Learning layer.

Delivers an article-level feature table (one row per article), joined by Person B
on `article_id`. This package never imports `market/`; the layers meet only at the
Parquet tables and shared constants in `config.py`.

MODULES AS BUILT, in pipeline order. The driver that runs them end to end is
notes/stage4-scripts/run_full_pipeline.py; see ARCHITECTURE.md §2 for the data flow
and which artifact each stage writes.

    entity_filter.py  sentence split, alias/entity tagging, anaphora, coref mapping,
                      boilerplate flagging  -> the 12-column SENTENCE_COLUMNS schema
    coref.py          fastcoref wrapper + cluster cache (backend for entity_filter)
    sentiment.py      FinBERT scoring, headline scoring, article-level aggregation,
                      and analyze() -- one code path for batch AND the live demo
    absa.py           aspect-based sentiment (DeBERTa), incl. the name substitution
                      that makes (text, aspect) pairs work on resolved anaphors
    fusion.py         combines FinBERT + ABSA into the shipped per-sentence score
                      (see CONF_FLOOR) and its article-level aggregates
    coref_judge.py    local Qwen2.5-7B referent-verification judge (llama.cpp)
    coref_eval.py     the harness that measures a judge against hand labels;
                      gates coref_judge

    company_registry.py  SHELVED, untracked, imported by nothing. A sourced
                      exchange-listing alias table, measured net harmful.
                      See notes/removed-mechanisms.md §4 -- the decision is open.

Every model-backed module degrades gracefully: is_available() returns False and the
pipeline continues with that stage's columns absent, rather than raising.
"""
