"""Run the referent-verification judge over EVERY coref-resolved sentence in the corpus.

Population: mentions_target & ~is_boilerplate & resolved_by_coref -- the sentences
the judge would actually gate in production. Boilerplate is excluded because
needs_score() already drops it, so judging it would be spent CPU with no effect.

CHUNKED AND RESUMABLE. ~3.5k sentences at ~4.9 s/row is ~4.6 hours, long enough
that a crash, a reboot or a killed terminal is a real risk. Verdicts are flushed
to the on-disk cache every CHUNK rows, and the cache is keyed
(article_id, sent_idx, target, model_id, prompt_version) with merge-never-replace
semantics -- so re-running this script picks up exactly where it left off and
never re-judges a row it has already answered.

Writes data/interim/judge_corpus_verdicts.parquet at the end: one row per judged
sentence with its channel, so the analysis can be redone without the model.
"""
import time

import pandas as pd
from loguru import logger

from stock_predictor.config import PRIMARY_TICKER
from stock_predictor.text.coref_eval import (
    CACHE_COLUMNS,
    build_context,
    load_judge_cache,
    save_judge_cache,
)
from stock_predictor.text.coref_judge import PROMPT_VERSION, make_judge
from stock_predictor.text.fusion import provenance_channel

CHUNK = 100
MODEL_ID = "qwen2.5-7b-instruct-q4km"

sentences = pd.read_parquet("data/sentences.parquet")
articles = pd.read_parquet("data/articles.parquet")
sentences["__channel"] = provenance_channel(sentences)

pop = sentences[
    sentences["mentions_target"].fillna(False)
    & ~sentences["is_boilerplate"].fillna(False)
    & sentences["resolved_by_coref"].fillna(False)
].copy()
logger.info(
    f"Judge population: {len(pop)} coref sentences "
    f"({int((pop['__channel'] == 'coref_span').sum())} span, "
    f"{int((pop['__channel'] == 'coref_nospan').sum())} no-span) "
    f"across {pop['article_id'].nunique()} articles"
)

head_col = "headline" if "headline" in articles.columns else "title"
headlines = articles.set_index("article_id")[head_col].to_dict()
by_article = {aid: g.sort_values("sent_idx") for aid, g in sentences.groupby("article_id")}

cache = load_judge_cache()
answered = {
    (r.article_id, r.sent_idx)
    for r in cache[
        (cache["model_id"] == MODEL_ID) & (cache["prompt_version"] == PROMPT_VERSION)
    ].itertuples(index=False)
}
todo = pop[~pop.set_index(["article_id", "sent_idx"]).index.isin(answered)]
logger.info(f"{len(answered)} already cached under {PROMPT_VERSION}; {len(todo)} to judge")

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
    fresh.append(
        {
            "article_id": int(row.article_id),
            "sent_idx": int(row.sent_idx),
            "target": PRIMARY_TICKER,
            "model_id": MODEL_ID,
            "prompt_version": PROMPT_VERSION,
            "answer": judge(ctx),
        }
    )

    if i % CHUNK == 0 or i == len(todo):
        # Merge, never replace -- `fresh` holds only this run's new verdicts.
        cache = load_judge_cache()
        save_judge_cache(
            pd.concat([cache, pd.DataFrame(fresh, columns=CACHE_COLUMNS)], ignore_index=True)
        )
        fresh = []
        rate = (time.time() - start) / i
        logger.info(
            f"{i}/{len(todo)} judged ({i/len(todo):.1%}), {rate:.2f}s/row, "
            f"~{(len(todo) - i) * rate / 3600:.1f}h remaining"
        )

# Final artifact: verdicts joined onto the population, channel included.
cache = load_judge_cache()
verdicts = cache[
    (cache["model_id"] == MODEL_ID) & (cache["prompt_version"] == PROMPT_VERSION)
]
out = pop[["article_id", "sent_idx", "__channel", "text"]].merge(
    verdicts[["article_id", "sent_idx", "answer"]], on=["article_id", "sent_idx"], how="left"
)
out = out.rename(columns={"__channel": "channel"})
out.to_parquet("data/interim/judge_corpus_verdicts.parquet", index=False)
logger.info(f"Wrote data/interim/judge_corpus_verdicts.parquet with {len(out)} rows")
print(out.groupby(["channel", "answer"]).size().to_string())
