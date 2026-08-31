"""Live text-layer demo: score one article's sentiment toward a chosen company.

This is a showcase of the text pipeline, not a price predictor. The shipped model
is session-level and needs market features and a whole session's articles, none of
which a single pasted article carries. What this demonstrates is the layer the
project's work went into: sentiment scored *toward the target company* rather than
in general, with the sentences that are actually about the target picked out.

It runs the fast path, `sentiment.analyze` (entity tagging plus FinBERT), not the
full coreference, ABSA and judge pipeline the batch corpus is built with, so it
stays interactive.

Run with:  streamlit run app/streamlit_app.py
"""

import pandas as pd
import streamlit as st

from stock_predictor.config import COMPANIES
from stock_predictor.text import entity_filter, sentiment

# A short Tesla example, so the demo does something on first load without the user
# having to find an article first.
EXAMPLE_TICKER = "TSLA"
EXAMPLE_HEADLINE = "Tesla deliveries beat expectations as Model Y demand recovers"
EXAMPLE_BODY = (
    "Tesla reported stronger-than-expected quarterly deliveries on Wednesday. "
    "The company said Model Y demand recovered sharply in China. "
    "Analysts had feared a weaker quarter after price cuts squeezed margins. "
    "Elon Musk called the results a turning point for the year. "
    "Ford, meanwhile, warned that its own EV losses would widen. "
    "The broader market was little changed on the day."
)


def _signed(pos, neg) -> float:
    """The signed sentiment used throughout the project: positive is favourable."""
    return float(pos) - float(neg)


def score_article(ticker: str, headline: str, body: str) -> dict:
    """Run the fast text path on one article. Pure, so it can be tested headlessly.

    Returns the article-level aggregate from `sentiment.analyze`, plus a per-sentence
    table showing which sentences were tagged as about the target and their FinBERT
    scores. Both come from the same public functions the batch pipeline uses.
    """
    body = (body or "").strip()
    headline = (headline or "").strip()

    article = sentiment.analyze(body, ticker, headline=headline or None)

    sentences = entity_filter.split_sentences([body])[0] if body else []
    tagged = entity_filter.tag_sentences("__demo__", sentences, ticker)
    if len(tagged):
        # only_relevant=False so every sentence is scored and shown, not just the
        # target ones; identical scoring to the batch path otherwise.
        scored = sentiment.score_sentence_table(tagged, only_relevant=False)
        scored["sentiment"] = scored["pos"] - scored["neg"]
        table = scored[["text", "mentions_target", "mentions_ceo", "sentiment"]].rename(
            columns={
                "text": "sentence",
                "mentions_target": "about_target",
                "mentions_ceo": "about_ceo",
            }
        )
    else:
        table = pd.DataFrame(columns=["sentence", "about_target", "about_ceo", "sentiment"])

    return {"article": article, "sentences": table}


def _company_label(ticker: str) -> str:
    names = COMPANIES.get(ticker, {}).get("names") or [ticker]
    return f"{names[0]} ({ticker})"


def main() -> None:
    st.set_page_config(page_title="Sentiment toward the company", page_icon="📈", layout="wide")
    st.title("Sentiment toward the company")
    st.caption(
        "A live demo of the text layer: sentiment scored toward a chosen company rather "
        "than in general. This is not a price predictor; the shipped model is "
        "session-level and needs market data a single article does not carry."
    )

    tickers = sorted(COMPANIES)
    default_idx = tickers.index(EXAMPLE_TICKER) if EXAMPLE_TICKER in tickers else 0

    with st.form("article"):
        ticker = st.selectbox(
            "Target company", tickers, index=default_idx, format_func=_company_label
        )
        headline = st.text_input("Headline", value=EXAMPLE_HEADLINE)
        body = st.text_area("Article body", value=EXAMPLE_BODY, height=220)
        submitted = st.form_submit_button("Analyze")

    if not submitted:
        return
    if not body.strip():
        st.warning("Paste an article body to analyze.")
        return

    with st.spinner("Loading FinBERT and scoring (first run downloads the model)…"):
        result = score_article(ticker, headline, body)

    article = result["article"]
    company = _company_label(ticker)

    n_total = int(article.get("n_total_sents") or 0)
    n_entity = int(article.get("n_entity_sents") or 0)
    share = article.get("entity_share") or 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentences", n_total)
    c2.metric(f"About {ticker}", n_entity)
    c3.metric("Relevance", f"{share:.0%}")
    c4.metric("Mentions CEO", "yes" if article.get("has_ceo_mention") else "no")

    st.subheader("Sentiment toward the company")
    if n_entity > 0:
        body_sent = _signed(article["sent_entity_pos"], article["sent_entity_neg"])
        st.metric(f"Body sentiment toward {company}", f"{body_sent:+.3f}")
    else:
        st.info(f"No sentence in the body was about {company}, so there is no body sentiment.")

    cols = st.columns(2)
    if pd.notna(article.get("sent_headline_pos")):
        head_sent = _signed(article["sent_headline_pos"], article["sent_headline_neg"])
        cols[0].metric("Headline sentiment", f"{head_sent:+.3f}")
    if article.get("has_ceo_mention") and pd.notna(article.get("sent_ceo_pos")):
        ceo_sent = _signed(article["sent_ceo_pos"], article["sent_ceo_neg"])
        cols[1].metric("Sentiment about the CEO", f"{ceo_sent:+.3f}")

    st.caption(
        "Positive is favourable to the company. Score is FinBERT `pos - neg`; the full "
        "batch pipeline additionally fuses this with an aspect-based model and gates it "
        "through a referent-verification judge."
    )

    st.subheader("Per-sentence breakdown")
    st.caption("Rows flagged `about_target` are the ones the body sentiment averages over.")
    st.dataframe(
        result["sentences"].style.format({"sentiment": "{:+.3f}"}),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
