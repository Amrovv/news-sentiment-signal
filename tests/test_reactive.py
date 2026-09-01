import pandas as pd
import pytest

from news_sentiment.text.entity_filter import SENTENCE_COLUMNS
from news_sentiment.text.reactive import (
    aggregate_reactive_features,
    classify_reactive_sentence,
    tag_reactive_sentences,
)


def _sent_row(article_id, sent_idx, text, mentions_target, is_boilerplate=False):
    return {
        "article_id": article_id,
        "sent_idx": sent_idx,
        "text": text,
        "mentions_target": mentions_target,
        "mentions_ceo": False,
        "resolved_by_coref": False,
        "is_boilerplate": is_boilerplate,
        "char_len": len(text),
    }


def test_classify_reactive_sentence_strong_hit():
    res = classify_reactive_sentence("Tesla shares rose 8% after the earnings call.")
    assert res.is_reactive == 1
    assert res.score >= 1.0
    assert res.hits


def test_classify_reactive_sentence_no_hit():
    res = classify_reactive_sentence("Tesla plans to open a new factory in Mexico.")
    assert res.is_reactive == 0
    assert res.score == 0.0
    assert res.hits == []


def test_classify_reactive_sentence_weak_only_below_threshold():
    # A single weak hit (0.5) sits below the default threshold of 1.0.
    res = classify_reactive_sentence("Investors reacted cautiously to the announcement.")
    assert res.score == 0.5
    assert res.is_reactive == 0


def test_tag_reactive_sentences_matches_schema_and_does_not_filter():
    df = pd.DataFrame(
        {col: [] for col in SENTENCE_COLUMNS}
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "article_id": [1, 1],
                    "sent_idx": [0, 1],
                    "text": [
                        "Tesla stock surged 12% on record deliveries.",
                        "The company also announced a new charging network.",
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

    tagged = tag_reactive_sentences(df)

    assert len(tagged) == len(df)
    assert {"is_reactive", "reactive_score", "reactive_hits"}.issubset(tagged.columns)
    assert tagged.loc[0, "is_reactive"] == 1
    assert tagged.loc[1, "is_reactive"] == 0
    for col in SENTENCE_COLUMNS:
        assert col in tagged.columns


def test_tag_reactive_sentences_empty_frame():
    df = pd.DataFrame({col: [] for col in SENTENCE_COLUMNS})
    tagged = tag_reactive_sentences(df)
    assert len(tagged) == 0
    assert {"is_reactive", "reactive_score", "reactive_hits"}.issubset(tagged.columns)


def test_aggregate_reactive_features_means_over_target_sentences_only():
    rows = [
        _sent_row(1, 0, "Tesla shares surged 10% on record deliveries.", True),
        _sent_row(1, 1, "The company also opened a new factory.", True),
        # Not about the target -- must not count toward article 1's reactivity.
        _sent_row(1, 2, "Meanwhile, Ford's shares plunged 8% on a recall.", False),
    ]
    df = pd.DataFrame(rows)
    tagged = tag_reactive_sentences(df)
    result = aggregate_reactive_features(tagged)

    row = result.iloc[0]
    assert row["article_id"] == 1
    assert row["n_reactive_sents"] == 1
    assert row["reactive_share"] == pytest.approx(0.5)
    assert row["reactive_mean"] == pytest.approx((1.0 + 0.0) / 2)
    assert row["reactive_max"] == pytest.approx(1.0)


def test_aggregate_reactive_features_excludes_boilerplate():
    rows = [
        _sent_row(2, 0, "Tesla stock jumped 15% today.", True, is_boilerplate=True),
        _sent_row(2, 1, "Deliveries also rose this quarter.", True),
    ]
    df = pd.DataFrame(rows)
    tagged = tag_reactive_sentences(df)
    result = aggregate_reactive_features(tagged)

    row = result.iloc[0]
    assert row["n_reactive_sents"] == 0
    assert row["reactive_mean"] == pytest.approx(0.0)


def test_aggregate_reactive_features_empty_target_population_is_nan_not_zero():
    rows = [
        _sent_row(3, 0, "BYD announced a new plant.", False),
    ]
    df = pd.DataFrame(rows)
    tagged = tag_reactive_sentences(df)
    result = aggregate_reactive_features(tagged)

    row = result.iloc[0]
    assert row["n_reactive_sents"] == 0
    assert pd.isna(row["reactive_share"])
    assert pd.isna(row["reactive_mean"])
    assert pd.isna(row["reactive_max"])


def test_aggregate_reactive_features_headline_kept_separate_from_body_mean():
    rows = [
        _sent_row(4, 0, "The company opened a new office.", True),  # not reactive
    ]
    df = pd.DataFrame(rows)
    tagged = tag_reactive_sentences(df)
    headlines_df = pd.DataFrame(
        {"article_id": [4], "headline": ["Tesla shares soar 12% on delivery beat"]}
    )
    result = aggregate_reactive_features(tagged, headlines_df=headlines_df)

    row = result.iloc[0]
    assert row["reactive_mean"] == pytest.approx(0.0)
    assert row["reactive_headline"] >= 1.0


def test_aggregate_reactive_features_no_headlines_df_gives_nan_headline_score():
    rows = [_sent_row(5, 0, "Tesla shares rose 5%.", True)]
    df = pd.DataFrame(rows)
    tagged = tag_reactive_sentences(df)
    result = aggregate_reactive_features(tagged)

    assert pd.isna(result.iloc[0]["reactive_headline"])


def test_classify_reactive_sentence_matches_plural_stocks():
    # "stock" required the exact singular before this fix; "stocks jumped 476%"
    # (a real miss found against the hand-labelled sample) fell through it.
    res = classify_reactive_sentence("Her stocks jumped 476% since joining Congress.")
    assert res.is_reactive == 1


def test_classify_reactive_sentence_excludes_market_share():
    # "market share" is a business metric, not a price move -- the shares?
    # pattern must not fire just because a move verb happens to follow it.
    res = classify_reactive_sentence("The company's market share fell in Europe.")
    assert res.score == 0.0
    assert res.hits == []


def test_classify_reactive_sentence_new_move_verbs():
    for text in [
        "Tech stocks see a big rebound on Powell speech.",
        "Tesla's stock just bounced back in a big way.",
        "10 stocks rocketing after Powell's dovish shift.",
    ]:
        res = classify_reactive_sentence(text)
        assert res.score > 0.0, text


def test_classify_reactive_sentence_outperform_without_market_suffix():
    # Previously required "outperformed the market/S&P/Nasdaq" verbatim;
    # loosened since the weak tier's own bar is "suggestive, not certain".
    res = classify_reactive_sentence("Overseas markets outperformed US stocks YTD.")
    assert res.score >= 0.5


def test_classify_reactive_sentence_direction_without_magnitude_is_weak():
    res = classify_reactive_sentence("Tesla stock is up, what you need to know.")
    assert res.score == 0.5
    assert res.is_reactive == 0


def test_aggregate_reactive_features_requires_tagging_first():
    df = pd.DataFrame([_sent_row(6, 0, "Tesla shares rose 5%.", True)])
    with pytest.raises(ValueError, match="tag_reactive_sentences"):
        aggregate_reactive_features(df)
