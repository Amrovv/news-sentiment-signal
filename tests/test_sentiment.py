import pandas as pd
import pytest

from stock_predictor.config import LEAD_SENTENCE_WINDOW
from stock_predictor.text.sentiment import (
    aggregate_article_features,
    analyze,
    hash_text,
    load_cache,
    save_cache,
    score_sentences,
)

# ---------------------------------------------------------------------------
# hash_text
# ---------------------------------------------------------------------------


def test_hash_text_deterministic():
    h1 = hash_text("Tesla surges on strong deliveries.")
    h2 = hash_text("Tesla surges on strong deliveries.")
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 16


def test_hash_text_differs_for_different_text():
    assert hash_text("Tesla surges.") != hash_text("Tesla plunges.")


def test_hash_text_strips_whitespace():
    assert hash_text("Tesla surges.") == hash_text("  Tesla surges.  ")


# ---------------------------------------------------------------------------
# load_cache / save_cache round trip
# ---------------------------------------------------------------------------


def test_load_cache_missing_file_returns_empty(tmp_path):
    path = tmp_path / "does_not_exist.parquet"
    df = load_cache(path)
    assert list(df.columns) == ["text_hash", "pos", "neg", "neu"]
    assert len(df) == 0


def test_save_and_load_cache_round_trip(tmp_path):
    path = tmp_path / "nested" / "cache.parquet"
    cache_df = pd.DataFrame(
        {
            "text_hash": ["aaaa", "bbbb"],
            "pos": [0.7, 0.1],
            "neg": [0.1, 0.8],
            "neu": [0.2, 0.1],
        }
    )
    save_cache(cache_df, path=path)
    assert path.exists()

    loaded = load_cache(path)
    assert len(loaded) == 2
    assert set(loaded["text_hash"]) == {"aaaa", "bbbb"}


def test_save_cache_dedups_on_text_hash_keep_last(tmp_path):
    path = tmp_path / "cache.parquet"
    cache_df = pd.DataFrame(
        {
            "text_hash": ["aaaa", "aaaa"],
            "pos": [0.1, 0.9],
            "neg": [0.8, 0.05],
            "neu": [0.1, 0.05],
        }
    )
    save_cache(cache_df, path=path)
    loaded = load_cache(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["pos"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# score_sentences: cache-hit behavior (no model required)
# ---------------------------------------------------------------------------


def test_score_sentences_all_cache_hits_requires_no_model():
    texts = ["Tesla beat earnings expectations.", "The firm raised its outlook."]
    cache_df = pd.DataFrame(
        {
            "text_hash": [hash_text(t) for t in texts],
            "pos": [0.8, 0.6],
            "neg": [0.1, 0.2],
            "neu": [0.1, 0.2],
        }
    )
    # model=None, tokenizer=None -- must not attempt to load a real model.
    result = score_sentences(texts, model=None, tokenizer=None, cache_df=cache_df)

    assert len(result) == 2
    assert set(result["text_hash"]) == set(cache_df["text_hash"])
    for _, row in result.iterrows():
        match = cache_df[cache_df["text_hash"] == row["text_hash"]].iloc[0]
        assert row["pos"] == pytest.approx(match["pos"])
        assert row["neg"] == pytest.approx(match["neg"])
        assert row["neu"] == pytest.approx(match["neu"])


def test_score_sentences_skips_cached_hashes_dedupes_input():
    # Duplicate text should collapse to a single unique hash internally.
    texts = ["Tesla beat earnings.", "Tesla beat earnings.", "Tesla beat earnings."]
    cache_df = pd.DataFrame(
        {
            "text_hash": [hash_text(texts[0])],
            "pos": [0.9],
            "neg": [0.05],
            "neu": [0.05],
        }
    )
    result = score_sentences(texts, model=None, tokenizer=None, cache_df=cache_df)
    assert len(result) == 1
    assert result.iloc[0]["pos"] == pytest.approx(0.9)


def test_score_sentences_empty_cache_df_but_no_texts_needing_score():
    # Edge case: no input texts at all.
    result = score_sentences([], model=None, tokenizer=None, cache_df=None)
    assert list(result.columns) == ["text_hash", "pos", "neg", "neu"]
    assert len(result) == 0


# ---------------------------------------------------------------------------
# score_sentences: real model, end-to-end (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_score_sentences_real_model_end_to_end():
    """Exercises the actual FinBERT model. Slow (~10-30s incl. download on
    first run / model load each run) -- this is the one required real-model
    test proving score_sentences() actually calls FinBERT correctly."""
    texts = [
        "Tesla shares surged after record delivery numbers beat estimates.",
        "Tesla shares plunged after the company missed delivery targets badly.",
        "Tesla will report its quarterly earnings next Tuesday.",
    ]
    result = score_sentences(texts, cache_df=None)

    assert len(result) == 3
    for col in ("pos", "neg", "neu"):
        assert col in result.columns

    # Probabilities must be valid and sum to ~1 per row.
    sums = result["pos"] + result["neg"] + result["neu"]
    for s in sums:
        assert s == pytest.approx(1.0, abs=1e-3)

    by_hash = {row.text_hash: row for row in result.itertuples(index=False)}
    pos_row = by_hash[hash_text(texts[0])]
    neg_row = by_hash[hash_text(texts[1])]

    # Sanity check on direction, not exact values.
    assert pos_row.pos > pos_row.neg
    assert neg_row.neg > neg_row.pos


# ---------------------------------------------------------------------------
# aggregate_article_features (no model needed -- pos/neg/neu pre-filled)
# ---------------------------------------------------------------------------


def _sent_row(article_id, sent_idx, text, mentions_target, mentions_other, pos, neg, neu):
    return {
        "article_id": article_id,
        "sent_idx": sent_idx,
        "text": text,
        "mentions_target": mentions_target,
        "mentions_other": mentions_other,
        "mentions_ceo": False,
        "resolved_by_anaphora": False,
        "char_len": len(text),
        "pos": pos,
        "neg": neg,
        "neu": neu,
    }


def test_aggregate_article_features_basic_means_and_counts():
    rows = [
        _sent_row(1, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
        _sent_row(1, 1, "Tesla stock dropped later.", True, False, 0.2, 0.7, 0.1),
        _sent_row(1, 2, "BYD also reported results.", False, True, 0.5, 0.3, 0.2),
    ]
    df = pd.DataFrame(rows)
    result = aggregate_article_features(df)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["article_id"] == 1
    assert row["n_entity_sents"] == 2
    assert row["n_other_sents"] == 1
    assert row["n_total_sents"] == 3
    assert row["entity_share"] == pytest.approx(2 / 3)
    assert row["sent_entity_pos"] == pytest.approx((0.8 + 0.2) / 2)
    assert row["sent_entity_neg"] == pytest.approx((0.1 + 0.7) / 2)
    assert row["sent_other_mean_pos"] == pytest.approx(0.5)
    assert row["article_length"] == sum(r["char_len"] for r in rows)


def test_aggregate_article_features_zero_target_sentences_is_nan_not_zero():
    rows = [
        _sent_row(2, 0, "BYD announced a new plant.", False, True, 0.5, 0.3, 0.2),
    ]
    df = pd.DataFrame(rows)
    result = aggregate_article_features(df)

    row = result.iloc[0]
    assert row["n_entity_sents"] == 0
    assert pd.isna(row["sent_entity_pos"])
    assert pd.isna(row["sent_entity_neg"])
    assert pd.isna(row["sent_entity_neu"])
    assert pd.isna(row["sent_entity_maxmag_pos"])
    assert row["entity_share"] == 0


def test_aggregate_article_features_maxmag_selects_largest_magnitude():
    rows = [
        # |pos-neg| = 0.1
        _sent_row(3, 0, "Tesla had a modest day.", True, False, 0.45, 0.35, 0.2),
        # |pos-neg| = 0.9 -- this should win
        _sent_row(3, 1, "Tesla crashed hard on bad news.", True, False, 0.05, 0.95, 0.0),
        # |pos-neg| = 0.3
        _sent_row(3, 2, "Tesla recovered somewhat.", True, False, 0.6, 0.3, 0.1),
    ]
    df = pd.DataFrame(rows)
    result = aggregate_article_features(df)
    row = result.iloc[0]

    assert row["sent_entity_maxmag_pos"] == pytest.approx(0.05)
    assert row["sent_entity_maxmag_neg"] == pytest.approx(0.95)
    assert row["sent_entity_maxmag_neu"] == pytest.approx(0.0)


def test_aggregate_article_features_lead_window():
    # LEAD_SENTENCE_WINDOW sentences (idx 0..N-1) are "lead"; put one target
    # sentence inside the window and one target sentence just after it.
    rows = [
        _sent_row(4, 0, "Tesla opened strong today.", True, False, 0.9, 0.05, 0.05),
        _sent_row(
            4,
            LEAD_SENTENCE_WINDOW,
            "Tesla later faced headwinds late in the piece.",
            True,
            False,
            0.1,
            0.8,
            0.1,
        ),
    ]
    df = pd.DataFrame(rows)
    result = aggregate_article_features(df)
    row = result.iloc[0]

    # Only the sent_idx=0 sentence is within the lead window.
    assert row["sent_entity_lead_pos"] == pytest.approx(0.9)
    assert row["sent_entity_lead_neg"] == pytest.approx(0.05)
    # But the overall entity mean includes both sentences.
    assert row["sent_entity_pos"] == pytest.approx((0.9 + 0.1) / 2)


def test_aggregate_article_features_lead_window_empty_is_nan():
    rows = [
        _sent_row(
            5,
            LEAD_SENTENCE_WINDOW + 1,
            "Tesla mentioned late in the article.",
            True,
            False,
            0.5,
            0.3,
            0.2,
        ),
    ]
    df = pd.DataFrame(rows)
    result = aggregate_article_features(df)
    row = result.iloc[0]
    assert pd.isna(row["sent_entity_lead_pos"])


def test_aggregate_article_features_headline_join():
    rows = [
        _sent_row(6, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
    ]
    df = pd.DataFrame(rows)
    headline_scores = pd.DataFrame(
        {"article_id": [6], "pos": [0.95], "neg": [0.02], "neu": [0.03]}
    )
    result = aggregate_article_features(df, headline_scores=headline_scores)
    row = result.iloc[0]
    assert row["sent_headline_pos"] == pytest.approx(0.95)
    assert row["sent_headline_neg"] == pytest.approx(0.02)


def test_aggregate_article_features_no_headline_scores_is_nan():
    rows = [
        _sent_row(7, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
    ]
    df = pd.DataFrame(rows)
    result = aggregate_article_features(df, headline_scores=None)
    row = result.iloc[0]
    assert pd.isna(row["sent_headline_pos"])
    assert pd.isna(row["sent_headline_neg"])
    assert pd.isna(row["sent_headline_neu"])


def test_aggregate_article_features_multiple_articles_independent():
    rows = [
        _sent_row(1, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
        _sent_row(2, 0, "Ford reported a slow quarter.", False, True, 0.2, 0.6, 0.2),
    ]
    df = pd.DataFrame(rows)
    result = aggregate_article_features(df)
    assert len(result) == 2
    assert set(result["article_id"]) == {1, 2}


# ---------------------------------------------------------------------------
# analyze(): single-article entry point (real model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_analyze_end_to_end():
    article_text = (
        "Tesla reported record deliveries this quarter, beating analyst estimates. "
        "The automaker also raised its full-year guidance after strong demand. "
        "BYD, a rival automaker, reported softer sales in the same period."
    )
    headline = "Tesla smashes delivery records, raises guidance"

    result = analyze(article_text, ticker="TSLA", headline=headline)

    assert isinstance(result, dict)
    assert result["n_entity_sents"] >= 1
    assert result["n_other_sents"] >= 1
    assert result["n_total_sents"] >= 2
    assert 0 <= result["entity_share"] <= 1
    assert not pd.isna(result["sent_entity_pos"])
    # Headline was provided, so headline sentiment must be populated.
    assert not pd.isna(result["sent_headline_pos"])
    assert result["article_length"] > 0


@pytest.mark.slow
def test_analyze_no_target_sentences_gives_nan_entity_scores():
    article_text = "BYD reported strong sales growth in China this quarter."
    result = analyze(article_text, ticker="TSLA", headline=None)

    assert result["n_entity_sents"] == 0
    assert pd.isna(result["sent_entity_pos"])
    assert pd.isna(result["sent_headline_pos"])
