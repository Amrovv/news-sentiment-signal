import pytest

from stock_predictor.config import MIN_SENT_CHARS
from stock_predictor.text.entity_filter import (
    resolve_anaphora,
    split_sentences,
    tag_sentences,
)

TICKER = "TSLA"


@pytest.fixture(scope="module")
def nlp():
    import spacy

    from stock_predictor.config import SPACY_MODEL

    return spacy.load(SPACY_MODEL, exclude=["ner", "lemmatizer"])


def test_basic_target_mention():
    df = tag_sentences(1, ["Tesla surges while rivals stumble."], TICKER)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["mentions_target"] == True
    assert row["mentions_other"] == False
    assert row["resolved_by_anaphora"] == False


def test_other_company_excluded_from_target():
    # Sentence 1 is about the target (TSLA), sentence 2 is about BYD only.
    # Key behavior under test: symmetry -- the BYD sentence must NOT be
    # counted as a TSLA target mention, but must be flagged as "other".
    sentences = [
        "Tesla delivered record numbers this quarter.",
        "BYD announced a new plant in Hungary this week.",
    ]
    df = tag_sentences(2, sentences, TICKER)

    tesla_row = df.iloc[0]
    byd_row = df.iloc[1]

    assert tesla_row["mentions_target"] == True
    assert tesla_row["mentions_other"] == False

    assert byd_row["mentions_target"] == False
    assert byd_row["mentions_other"] == True


@pytest.mark.parametrize(
    "sentence",
    [
        "The soft drink giant is expanding sales in Tokyo and Osaka this year.",
        "Analysts pointed to tsla-stock-analysis pages as unreliable clickbait sources.",
    ],
)
def test_ticker_substring_traps(sentence):
    # "KO" must not match inside "Tokyo"/"Osaka"-adjacent prose.
    ko_df = tag_sentences(3, [sentence], "KO")
    assert ko_df.iloc[0]["mentions_target"] == False

    # "TSLA" must not match a URL-fragment-like substring embedded in prose.
    tsla_df = tag_sentences(3, [sentence], "TSLA")
    assert tsla_df.iloc[0]["mentions_target"] == False


def test_anaphora_chain_tracks_most_recent_company():
    sentences = [
        "Tesla delivered record numbers.",
        "The company also raised prices.",
        "BYD announced a new plant.",
        "The firm said demand was strong.",
    ]
    df = resolve_anaphora(4, sentences, TICKER)

    # Sentence 1: explicit Tesla mention.
    assert df.iloc[0]["mentions_target"] == True
    assert df.iloc[0]["resolved_by_anaphora"] == False

    # Sentence 2: "The company" resolves to Tesla (last-named = target).
    assert df.iloc[1]["mentions_target"] == True
    assert df.iloc[1]["mentions_other"] == False
    assert df.iloc[1]["resolved_by_anaphora"] == True

    # Sentence 3: explicit BYD mention -> other, not target.
    assert df.iloc[2]["mentions_target"] == False
    assert df.iloc[2]["mentions_other"] == True
    assert df.iloc[2]["resolved_by_anaphora"] == False

    # Sentence 4: "The firm" resolves to BYD (last-named), NOT Tesla.
    assert df.iloc[3]["mentions_target"] == False
    assert df.iloc[3]["mentions_other"] == True
    assert df.iloc[3]["resolved_by_anaphora"] == True


def test_anaphora_decays_after_max_gap():
    # Long-article regression: an explicit mention many sentences back should
    # stop being attributed to generic anaphora once ANAPHORA_MAX_GAP is
    # exceeded, instead of drifting onto unrelated later content (the
    # dominant error mode found in the sentences.parquet hand-check --
    # see notebooks/a_entity_sentiment.ipynb section 4.3).
    from stock_predictor.config import ANAPHORA_MAX_GAP

    filler = [f"This is unrelated filler sentence number {i} about the weather." for i in range(ANAPHORA_MAX_GAP - 1)]
    sentences = ["Tesla delivered record numbers this quarter.", *filler, "The company said demand was strong."]
    df = resolve_anaphora(6, sentences, TICKER)

    # Sentence just inside the gap (gap == ANAPHORA_MAX_GAP) still resolves.
    assert df.iloc[-1]["mentions_target"] == True
    assert df.iloc[-1]["resolved_by_anaphora"] == True

    # One sentence further out, the antecedent has decayed.
    sentences_too_far = sentences + ["The firm reported strong results."]
    df2 = resolve_anaphora(6, sentences_too_far, TICKER)
    last = df2.iloc[-1]
    assert last["mentions_target"] == False
    assert last["mentions_other"] == False
    assert last["resolved_by_anaphora"] == False


def test_mentions_ceo_independent_of_mentions_target():
    # Design decision (documented in entity_filter.tag_sentences docstring):
    # person-tier aliases are ambiguous (e.g. "Musk" could mean SpaceX/X), so
    # mentions_ceo is tracked independently and never sets mentions_target on
    # its own. A sentence with "Musk" but no unambiguous/anaphoric Tesla
    # alias should have mentions_ceo=True and mentions_target=False.
    df = tag_sentences(5, ["Musk unveiled a new product at the event."], TICKER)
    row = df.iloc[0]
    assert row["mentions_ceo"] == True
    assert row["mentions_target"] == False

    # Sentence with both an unambiguous alias and a person alias: both flags
    # can be True simultaneously since they are independent.
    df2 = tag_sentences(5, ["Musk said Tesla deliveries hit a new record."], TICKER)
    row2 = df2.iloc[0]
    assert row2["mentions_ceo"] == True
    assert row2["mentions_target"] == True


def test_split_sentences_drops_short_scraper_residue(nlp):
    texts = ["Tesla stock rallied sharply today after strong earnings. Ad. Read more."]
    result = split_sentences(texts, nlp=nlp)
    assert len(result) == 1
    for sent in result[0]:
        assert len(sent) >= MIN_SENT_CHARS
    assert not any(sent in ("Ad.", "Read more.") for sent in result[0])


def test_split_sentences_no_false_splits_on_financial_abbreviations(nlp):
    text = (
        "Tesla posted Q3 2024 results, beating U.S. estimates. "
        "The company raised $1.2B in Q3. "
        "Tesla Inc. reaffirmed its guidance for the year."
    )
    result = split_sentences([text], nlp=nlp)
    assert len(result) == 1
    sentences = result[0]
    assert len(sentences) == 3
    assert sentences[0] == "Tesla posted Q3 2024 results, beating U.S. estimates."
    assert sentences[1] == "The company raised $1.2B in Q3."
    assert sentences[2] == "Tesla Inc. reaffirmed its guidance for the year."


def test_split_sentences_fixes_missing_space_after_period(nlp):
    text = "Tesla posted record deliveries.Musk said the results were strong across all regions."
    result = split_sentences([text], nlp=nlp)
    assert len(result) == 1
    sentences = result[0]
    assert len(sentences) == 2
    assert sentences[0] == "Tesla posted record deliveries."
    assert sentences[1] == "Musk said the results were strong across all regions."
