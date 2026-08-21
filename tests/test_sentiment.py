import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_predictor.config import LEAD_SENTENCE_WINDOW
from stock_predictor.text import sentiment as sentiment_module
from stock_predictor.text.fusion import AGGREGATED_VARIANTS, FUSION_AGGREGATIONS
from stock_predictor.text.sentiment import (
    ABSA_FEATURE_COLUMNS,
    FUSION_FEATURE_COLUMNS,
    aggregate_article_features,
    analyze,
    hash_text,
    load_cache,
    needs_score,
    save_cache,
    score_headlines,
    score_sentence_table,
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
# Cache persistence: saves must MERGE with the existing on-disk cache, never
# replace it. Regression cover for the notebook 2.1 failure mode where a full
# corpus run did score_sentence_table() (writing ~56k sentence entries) and
# then score_headlines(), whose save wiped every sentence entry because it
# persisted only the frame returned for the current call's texts.
#
# All of these are cache-hit-only calls (every input hash is pre-seeded in the
# cache), so no FinBERT model is ever loaded -- same trick as
# test_score_sentences_all_cache_hits_requires_no_model.
# ---------------------------------------------------------------------------


def _cache_frame(texts, pos=0.5, neg=0.3, neu=0.2):
    """Build a cache frame covering `texts`, so calls over them are pure hits."""
    return pd.DataFrame(
        {
            "text_hash": [hash_text(t) for t in texts],
            "pos": [pos] * len(texts),
            "neg": [neg] * len(texts),
            "neu": [neu] * len(texts),
        }
    )


def test_score_headlines_preserves_existing_cache_entries(tmp_path):
    path = tmp_path / "cache.parquet"
    sentence_texts = [f"Tesla sentence number {i}." for i in range(5)]
    headline = "Tesla smashes delivery records"

    save_cache(
        pd.concat([_cache_frame(sentence_texts), _cache_frame([headline], 0.9, 0.05, 0.05)]),
        path=path,
    )
    original_hashes = set(load_cache(path)["text_hash"])

    headlines_df = pd.DataFrame({"article_id": [1], "headline": [headline]})
    result = score_headlines(headlines_df, cache_path=path)

    assert result.iloc[0]["pos"] == pytest.approx(0.9)

    after = load_cache(path)
    assert set(after["text_hash"]) >= original_hashes
    for t in sentence_texts:
        assert hash_text(t) in set(after["text_hash"])


def test_score_sentence_table_preserves_unrelated_cache_entries(tmp_path):
    path = tmp_path / "cache.parquet"
    table_texts = ["Tesla beat estimates.", "Tesla raised guidance."]
    unrelated_texts = [f"Unrelated cached sentence {i}." for i in range(4)]

    save_cache(
        pd.concat([_cache_frame(table_texts, 0.7, 0.2, 0.1), _cache_frame(unrelated_texts)]),
        path=path,
    )

    sentences_df = pd.DataFrame(
        {
            "article_id": [1, 1],
            "sent_idx": [0, 1],
            "text": table_texts,
            # needs_score() schema: all rows relevant, so every text is scored
            # under the new only_relevant=True default.
            "mentions_target": [True, True],
            "mentions_ceo": [False, False],
        }
    )
    merged = score_sentence_table(sentences_df, cache_path=path)

    assert len(merged) == 2
    assert merged["pos"].tolist() == pytest.approx([0.7, 0.7])

    after_hashes = set(load_cache(path)["text_hash"])
    for t in unrelated_texts:
        assert hash_text(t) in after_hashes
    for t in table_texts:
        assert hash_text(t) in after_hashes


def test_full_corpus_sequence_headline_save_does_not_wipe_sentence_entries(tmp_path):
    """The exact real-world wipe sequence: score the sentence table, then score
    headlines. After the headline save, the sentence entries must survive."""
    path = tmp_path / "cache.parquet"
    sentence_texts = [f"Tesla sentence number {i}." for i in range(6)]
    headline = "Tesla smashes delivery records"

    # Seed everything so both calls are pure cache hits (no model load).
    save_cache(
        pd.concat([_cache_frame(sentence_texts), _cache_frame([headline], 0.9, 0.05, 0.05)]),
        path=path,
    )

    sentences_df = pd.DataFrame(
        {
            "article_id": [1] * len(sentence_texts),
            "sent_idx": list(range(len(sentence_texts))),
            "text": sentence_texts,
            # needs_score() schema: all rows relevant, so every text is scored.
            "mentions_target": [True] * len(sentence_texts),
            "mentions_ceo": [False] * len(sentence_texts),
        }
    )
    score_sentence_table(sentences_df, cache_path=path)

    after_sentences = set(load_cache(path)["text_hash"])
    assert all(hash_text(t) in after_sentences for t in sentence_texts)

    score_headlines(pd.DataFrame({"article_id": [1], "headline": [headline]}), cache_path=path)

    after_headlines = set(load_cache(path)["text_hash"])
    assert hash_text(headline) in after_headlines
    # The wipe bug showed up exactly here: only the headline hash survived.
    for t in sentence_texts:
        assert hash_text(t) in after_headlines, "headline save wiped sentence cache entries"


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


def _sent_row(
    article_id,
    sent_idx,
    text,
    mentions_target,
    _unused_other,   # other-company detection removed; slot kept so the many
                     # positional call sites below need no churn
    pos,
    neg,
    neu,
    mentions_ceo=False,
    is_boilerplate=False,
):
    return {
        "article_id": article_id,
        "sent_idx": sent_idx,
        "text": text,
        "mentions_target": mentions_target,
        "mentions_ceo": mentions_ceo,
        "resolved_by_coref": False,
        "is_boilerplate": is_boilerplate,
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
    assert row["n_total_sents"] == 3
    assert row["entity_share"] == pytest.approx(2 / 3)
    assert row["sent_entity_pos"] == pytest.approx((0.8 + 0.2) / 2)
    assert row["sent_entity_neg"] == pytest.approx((0.1 + 0.7) / 2)
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


def test_aggregate_boilerplate_excluded_from_means_and_totals():
    rows = [
        _sent_row(12, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
        _sent_row(
            12,
            1,
            "Image source: Getty Images.",
            True,
            False,
            0.0,
            0.9,
            0.1,
            is_boilerplate=True,
        ),
    ]
    result = aggregate_article_features(pd.DataFrame(rows))
    row = result.iloc[0]

    # Boilerplate row is out of the means, the entity count and the total.
    assert row["sent_entity_pos"] == pytest.approx(0.8)
    assert row["n_entity_sents"] == 1
    assert row["n_total_sents"] == 1
    assert row["entity_share"] == pytest.approx(1.0)
    # But it is counted and still contributes to the physical article length.
    assert row["n_boilerplate_sents"] == 1
    assert row["article_length"] == sum(r["char_len"] for r in rows)


def test_aggregate_missing_boilerplate_column_defaults_false():
    rows = [
        _sent_row(13, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
    ]
    df = pd.DataFrame(rows).drop(columns=["is_boilerplate"])
    result = aggregate_article_features(df)  # must not raise KeyError
    row = result.iloc[0]
    assert row["n_boilerplate_sents"] == 0
    assert row["sent_entity_pos"] == pytest.approx(0.8)


def test_aggregate_ceo_only_sentences():
    rows = [
        _sent_row(14, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
        # CEO mention WITH a target mention -> not a CEO-only sentence.
        _sent_row(
            14, 1, "Musk said Tesla is fine.", True, False, 0.5, 0.3, 0.2, mentions_ceo=True
        ),
        # CEO-only sentence.
        _sent_row(
            14, 2, "Musk unveiled a rocket.", False, False, 0.2, 0.6, 0.2, mentions_ceo=True
        ),
    ]
    result = aggregate_article_features(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row["n_ceo_sents"] == 1
    assert row["sent_ceo_pos"] == pytest.approx(0.2)
    assert row["sent_ceo_neg"] == pytest.approx(0.6)
    assert row["sent_ceo_neu"] == pytest.approx(0.2)


def test_aggregate_ceo_means_nan_when_no_ceo_only_sentences():
    rows = [
        _sent_row(15, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
    ]
    result = aggregate_article_features(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row["n_ceo_sents"] == 0
    assert pd.isna(row["sent_ceo_pos"])
    assert pd.isna(row["sent_ceo_neg"])
    assert pd.isna(row["sent_ceo_neu"])


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
# fus_* fusion columns wired into aggregate_article_features()
# ---------------------------------------------------------------------------


def _sent_row_with_absa(
    article_id, sent_idx, text, mentions_target, _unused_other, pos, neg, neu, absa_pos, absa_neg, absa_neu
):
    row = _sent_row(article_id, sent_idx, text, mentions_target, _unused_other, pos, neg, neu)
    row["absa_pos"] = absa_pos
    row["absa_neg"] = absa_neg
    row["absa_neu"] = absa_neu
    return row


def test_aggregate_article_features_includes_fusion_columns():
    rows = [
        _sent_row_with_absa(
            1, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1, 0.7, 0.2, 0.1
        ),
        _sent_row_with_absa(
            1, 1, "Tesla stock dropped later.", True, False, 0.2, 0.7, 0.1, 0.3, 0.6, 0.1
        ),
    ]
    df = pd.DataFrame(rows)
    result = aggregate_article_features(df)

    # One promoted variant x six aggregations. Derived rather than hardcoded,
    # so promoting or demoting a variant does not leave a stale literal here.
    assert len(FUSION_FEATURE_COLUMNS) == len(AGGREGATED_VARIANTS) * len(FUSION_AGGREGATIONS)
    for col in FUSION_FEATURE_COLUMNS:
        assert col in result.columns

    row = result.iloc[0]
    # Population is non-empty (2 target, non-boilerplate sentences with real
    # scores), so every promoted column should be a real number. Driven off the
    # constants so it exercises the whole family however many variants ship.
    for variant in AGGREGATED_VARIANTS:
        for agg in FUSION_AGGREGATIONS:
            assert not pd.isna(row[f"fus_{variant}_{agg}"])


def test_aggregate_article_features_fusion_nan_when_absa_absent():
    # No absa_pos/absa_neg/absa_neu columns at all -- fusion cannot be
    # computed. The whole fus_* family must be NaN and the function must not
    # raise (same soft feature-detect contract as absa_*).
    rows = [
        _sent_row(1, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
        _sent_row(1, 1, "Tesla stock dropped later.", True, False, 0.2, 0.7, 0.1),
    ]
    df = pd.DataFrame(rows)
    result = aggregate_article_features(df)

    assert len(FUSION_FEATURE_COLUMNS) == len(AGGREGATED_VARIANTS) * len(FUSION_AGGREGATIONS)
    for col in FUSION_FEATURE_COLUMNS:
        assert col in result.columns
        assert pd.isna(result.iloc[0][col])


# ---------------------------------------------------------------------------
# analyze(): single-article entry point (real model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_analyze_end_to_end(tmp_path):
    article_text = (
        "Tesla reported record deliveries this quarter, beating analyst estimates. "
        "The automaker also raised its full-year guidance after strong demand. "
        "BYD, a rival automaker, reported softer sales in the same period."
    )
    headline = "Tesla smashes delivery records, raises guidance"

    result = analyze(
        article_text, ticker="TSLA", headline=headline, cache_path=tmp_path / "cache.parquet"
    )

    assert isinstance(result, dict)
    assert result["n_entity_sents"] >= 1
    assert result["n_total_sents"] >= 2
    assert 0 <= result["entity_share"] <= 1
    assert not pd.isna(result["sent_entity_pos"])
    # Headline was provided, so headline sentiment must be populated.
    assert not pd.isna(result["sent_headline_pos"])
    assert result["article_length"] > 0

    # New schema flows through the single-article path too.
    for key in (
        "sent_ceo_pos",
        "sent_ceo_neg",
        "sent_ceo_neu",
        "n_boilerplate_sents",
        "n_ceo_sents",
    ):
        assert key in result
    # analyze() is single-article, so corpus-level boilerplate never fires.
    assert result["n_boilerplate_sents"] == 0


@pytest.mark.slow
def test_analyze_no_target_sentences_gives_nan_entity_scores(tmp_path):
    article_text = "BYD reported strong sales growth in China this quarter."
    result = analyze(
        article_text, ticker="TSLA", headline=None, cache_path=tmp_path / "cache.parquet"
    )

    assert result["n_entity_sents"] == 0
    assert pd.isna(result["sent_entity_pos"])
    assert pd.isna(result["sent_headline_pos"])


# ---------------------------------------------------------------------------
# needs_score(): the single source of truth for "does any feature read this
# sentence's FinBERT score?", shared by score_sentence_table() and
# aggregate_article_features() so the two cannot drift.
# ---------------------------------------------------------------------------


def test_needs_score_buckets():
    rows = [
        _sent_row(1, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
        # An other-company sentence. Was in-bucket while sent_other_mean_*
        # existed; nothing reads it now, so needs_score() must reject it.
        _sent_row(1, 1, "BYD reported results.", False, True, 0.5, 0.3, 0.2),
        _sent_row(1, 2, "Musk unveiled a rocket.", False, False, 0.2, 0.6, 0.2, mentions_ceo=True),
        _sent_row(1, 3, "Markets were quiet on Monday.", False, False, 0.3, 0.3, 0.4),
        _sent_row(
            1,
            4,
            "Image source: Getty Images.",
            True,
            False,
            0.0,
            0.9,
            0.1,
            is_boilerplate=True,
        ),
    ]
    mask = needs_score(pd.DataFrame(rows))

    assert mask.dtype == bool
    assert mask.tolist() == [True, False, True, False, False]


def test_needs_score_missing_boilerplate_column_defaults_false():
    rows = [
        _sent_row(1, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1),
        _sent_row(1, 1, "Markets were quiet.", False, False, 0.3, 0.3, 0.4),
    ]
    df = pd.DataFrame(rows).drop(columns=["is_boilerplate"])
    mask = needs_score(df)  # must not raise
    # Nothing is excluded on the boilerplate basis; only bucket membership decides.
    assert mask.tolist() == [True, False]


def test_needs_score_missing_mentions_column_raises():
    rows = [_sent_row(1, 0, "Tesla beat estimates.", True, False, 0.8, 0.1, 0.1)]
    df = pd.DataFrame(rows).drop(columns=["mentions_ceo"])
    with pytest.raises(ValueError, match="mentions_ceo"):
        needs_score(df)


def test_needs_score_empty_frame_returns_empty_bool_series():
    df = pd.DataFrame(
        columns=["mentions_target", "mentions_ceo", "is_boilerplate"]
    )
    mask = needs_score(df)
    assert len(mask) == 0
    assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Selective scoring (score_sentence_table(only_relevant=...)).
#
# Both scoring paths below are pure cache hits (every text's hash is pre-seeded
# in a tmp_path cache), so no FinBERT model is ever loaded -- same trick as
# test_score_sentences_all_cache_hits_requires_no_model.
# ---------------------------------------------------------------------------


def _bucket_coverage_rows():
    """A synthetic multi-article sentence table covering every bucket: target,
    other, CEO-only, neither, boilerplate, comparative and lead-window rows."""
    return [
        # Article 100: lead-window target, late target, other, neither.
        _sent_row(100, 0, "Tesla opened strong on deliveries.", True, False, 0.85, 0.1, 0.05),
        _sent_row(
            100,
            LEAD_SENTENCE_WINDOW + 2,
            "Tesla later gave back some of those gains.",
            True,
            False,
            0.15,
            0.75,
            0.10,
        ),
        _sent_row(100, 1, "BYD reported softer sales.", False, True, 0.30, 0.55, 0.15),
        _sent_row(100, 2, "The weather in Austin was mild.", False, False, 0.33, 0.33, 0.34),
        _sent_row(
            100,
            3,
            "Image source: Getty Images.",
            True,
            False,
            0.01,
            0.95,
            0.04,
            is_boilerplate=True,
        ),
        # Article 200: comparative target, CEO-only, CEO+target, boilerplate-other.
        _sent_row(
            200,
            0,
            "Micron has overtaken Tesla in market cap.",
            True,
            True,
            0.20,
            0.70,
            0.10,
        ),
        _sent_row(200, 1, "Tesla still leads on software.", True, False, 0.65, 0.20, 0.15),
        _sent_row(200, 2, "Musk unveiled a new rocket.", False, False, 0.40, 0.35, 0.25,
                  mentions_ceo=True),
        _sent_row(200, 3, "Musk said Tesla is fine.", True, False, 0.55, 0.25, 0.20,
                  mentions_ceo=True),
        _sent_row(
            200,
            4,
            "This article is for informational purposes only.",
            False,
            True,
            0.10,
            0.10,
            0.80,
            is_boilerplate=True,
        ),
        # Article 300: no target sentence at all, plus a scoreless filler row.
        _sent_row(300, 0, "Ford announced a new plant in Ohio.", False, True, 0.45, 0.30, 0.25),
        _sent_row(300, 1, "Nothing much happened afterwards.", False, False, 0.30, 0.40, 0.30),
    ]


def _seeded_table_and_cache(tmp_path):
    """Return (sentences_df without scores, cache_path) with every text's hash
    pre-seeded so both scoring paths are pure cache hits."""
    rows = _bucket_coverage_rows()
    df = pd.DataFrame(rows)
    cache = pd.DataFrame(
        {
            "text_hash": [hash_text(r["text"]) for r in rows],
            "pos": [r["pos"] for r in rows],
            "neg": [r["neg"] for r in rows],
            "neu": [r["neu"] for r in rows],
        }
    )
    path = tmp_path / "cache.parquet"
    save_cache(cache, path=path)
    return df.drop(columns=["pos", "neg", "neu"]), path


def test_filtered_and_full_scoring_produce_identical_aggregates(tmp_path):
    """THE GATE. Selective scoring may not change a single feature value.

    If someone adds a feature that reads a sentence bucket needs_score() does
    not keep, the filtered path will feed it NaN and this test fails loudly.
    """
    df, path = _seeded_table_and_cache(tmp_path)

    filtered = score_sentence_table(df, cache_path=path, only_relevant=True)
    full = score_sentence_table(df, cache_path=path, only_relevant=False)

    # The filtered path really did skip rows (otherwise this proves nothing).
    assert filtered["pos"].isna().sum() > 0
    assert full["pos"].isna().sum() == 0

    agg_filtered = aggregate_article_features(filtered)
    agg_full = aggregate_article_features(full)

    assert_frame_equal(agg_filtered, agg_full, check_exact=False)


def test_unscored_rows_are_nan_not_zero(tmp_path):
    df, path = _seeded_table_and_cache(tmp_path)
    scored = score_sentence_table(df, cache_path=path, only_relevant=True)

    mask = needs_score(df)
    unscored = scored[~mask.values]
    assert len(unscored) > 0
    for col in ("pos", "neg", "neu"):
        assert unscored[col].isna().all(), f"{col} must be NaN, not 0, for unscored rows"

    # Every relevant row IS scored.
    assert scored.loc[mask.values, ["pos", "neg", "neu"]].notna().all().all()


def test_score_sentence_table_only_relevant_skips_forward_passes(tmp_path):
    """only_relevant=True must not even hand the skipped texts to the scorer."""
    df, path = _seeded_table_and_cache(tmp_path)
    seen = {}

    import stock_predictor.text.sentiment as sentiment_module

    real = sentiment_module.score_sentences

    def spy(texts, *args, **kwargs):
        seen["n"] = len(texts)
        return real(texts, *args, **kwargs)

    sentiment_module.score_sentences = spy
    try:
        score_sentence_table(df, cache_path=path, only_relevant=True)
        n_filtered = seen["n"]
        score_sentence_table(df, cache_path=path, only_relevant=False)
        n_full = seen["n"]
    finally:
        sentiment_module.score_sentences = real

    assert n_full == len(df)
    assert n_filtered == int(needs_score(df).sum())
    assert n_filtered < n_full


@pytest.mark.slow
def test_analyze_unchanged_keys(tmp_path):
    """Selective scoring inside analyze() must not change its returned keys."""
    article_text = (
        "Tesla reported record deliveries this quarter, beating analyst estimates. "
        "The automaker also raised its full-year guidance after strong demand. "
        "BYD, a rival automaker, reported softer sales in the same period. "
        "The weather in Austin was unremarkable that week."
    )
    result = analyze(
        article_text,
        ticker="TSLA",
        headline="Tesla smashes delivery records",
        cache_path=tmp_path / "cache.parquet",
    )

    expected_keys = {
        "article_id",
        "sent_entity_pos",
        "sent_entity_neg",
        "sent_entity_neu",
        "sent_entity_maxmag_pos",
        "sent_entity_maxmag_neg",
        "sent_entity_maxmag_neu",
        "sent_entity_lead_pos",
        "sent_entity_lead_neg",
        "sent_entity_lead_neu",
        "sent_ceo_pos",
        "sent_ceo_neg",
        "sent_ceo_neu",
        "n_entity_sents",
        "n_boilerplate_sents",
        "n_ceo_sents",
        "n_total_sents",
        "entity_share",
        "article_length",
        "sent_headline_pos",
        "sent_headline_neg",
        "sent_headline_neu",
        # ABSA-parallel columns. analyze() does not run the aspect-based pass,
        # so these are present and all-NaN -- which is the contract: the
        # pipeline works with ABSA off.
        *ABSA_FEATURE_COLUMNS,
        # fus_* fusion columns: analyze() does not run ABSA either, so fusion
        # cannot be computed -- these are present and all-NaN for the same
        # reason the absa_* columns are.
        *FUSION_FEATURE_COLUMNS,
    }
    assert set(result.keys()) == expected_keys
    assert not pd.isna(result["sent_entity_pos"])
    assert all(pd.isna(result[c]) for c in ABSA_FEATURE_COLUMNS)
    assert all(pd.isna(result[c]) for c in FUSION_FEATURE_COLUMNS)


@pytest.mark.slow
def test_analyze_does_not_touch_the_default_cache(tmp_path, monkeypatch):
    """Regression test for analyze() unconditionally reading/writing
    config.SENTIMENT_CACHE_PATH via bare load_cache()/save_cache() calls with
    no path argument. Before the cache_path parameter was added, every test
    (and every interactive call) that exercised analyze() silently wrote
    synthetic sentences into data/interim/finbert_cache.parquet and bumped
    its mtime, mutating a real data file as a side effect of running tests.

    Hermetic by construction: rather than asserting on the real
    config.SENTIMENT_CACHE_PATH (whose existence and contents vary by
    machine), this monkeypatches the module attribute to a decoy path under
    tmp_path that nothing legitimately writes to, then asserts that path was
    never created. If analyze() ever regresses back to a bare, argument-less
    load_cache()/save_cache() call, any code path that re-reads the current
    module-level SENTIMENT_CACHE_PATH at call time would touch the decoy and
    fail this test; the explicit cache_path passed below is what must
    actually receive the write.
    """
    decoy_default_cache = tmp_path / "decoy_default_cache.parquet"
    monkeypatch.setattr(sentiment_module, "SENTIMENT_CACHE_PATH", decoy_default_cache)

    real_cache_path = tmp_path / "real_cache.parquet"
    article_text = "Tesla reported record deliveries this quarter, beating estimates."

    analyze(article_text, ticker="TSLA", cache_path=real_cache_path)

    assert not decoy_default_cache.exists(), (
        "analyze() wrote to the default SENTIMENT_CACHE_PATH even though an "
        "explicit cache_path was given"
    )
    assert real_cache_path.exists()
