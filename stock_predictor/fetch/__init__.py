"""Article collection: the Finnhub pull (`finnhub_pull`) and the per-article
scrape that recovers true source, true time, and clean body (`scrape`).

Ported from `notebooks/text/1.0-aw-corpus.ipynb` and
`notebooks/text/1.2-aw-clean-and-convert.ipynb`. Those notebooks are a
historical record of the original, one-off pull and are left as they were
run; they do not import from here. This package is where that logic now
lives for the re-engineered fetch going forward.
"""
