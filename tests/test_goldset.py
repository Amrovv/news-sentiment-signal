import pandas as pd
import pytest

from stock_predictor.text import goldset


def _row(
    article_id,
    sent_idx,
    text="A sentence about a company.",
    *,
    mentions_target=False,
    mentions_other=False,
    mentions_ceo=False,
    is_comparative=False,
    resolved_by_coref=False,
    resolved_by_anaphora=False,
    is_boilerplate=False,
    other_source="",
    pos=None,
    neg=None,
    neu=None,
    absa_pos=None,
    absa_neg=None,
    absa_neu=None,
    absa_text=None,
    absa_aspect=None,
):
    """One synthetic sentence row, spelled out by hand -- no real parquet
    files are read in this test module (per the task spec)."""
    return {
        "article_id": article_id,
        "sent_idx": sent_idx,
        "text": text,
        "mentions_target": mentions_target,
        "mentions_other": mentions_other,
        "mentions_ceo": mentions_ceo,
        "is_comparative": is_comparative,
        "resolved_by_coref": resolved_by_coref,
        "resolved_by_anaphora": resolved_by_anaphora,
        "is_boilerplate": is_boilerplate,
        "other_source": other_source,
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "absa_pos": absa_pos,
        "absa_neg": absa_neg,
        "absa_neu": absa_neu,
        "absa_text": absa_text,
        "absa_aspect": absa_aspect,
    }


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# assign_stratum
# ---------------------------------------------------------------------------


def test_comparative_and_coref_resolved_row_gets_comparative():
    # Highest priority: both is_comparative and resolved_by_coref are True,
    # comparative must win.
    df = _frame(
        [
            _row(
                1,
                0,
                mentions_target=True,
                mentions_other=True,
                is_comparative=True,
                resolved_by_coref=True,
            )
        ]
    )
    result = goldset.assign_stratum(df)
    assert result.iloc[0] == "comparative"


def test_boilerplate_row_gets_empty_stratum_regardless_of_flags():
    df = _frame(
        [
            _row(
                1,
                0,
                mentions_target=True,
                is_comparative=True,
                is_boilerplate=True,
            )
        ]
    )
    result = goldset.assign_stratum(df)
    assert result.iloc[0] == ""


def test_row_matching_no_bucket_gets_empty_stratum():
    df = _frame([_row(1, 0)])  # no mentions_* flags at all
    result = goldset.assign_stratum(df)
    assert result.iloc[0] == ""


def test_direct_stratum_for_explicit_unresolved_target_mention():
    df = _frame([_row(1, 0, mentions_target=True)])
    result = goldset.assign_stratum(df)
    assert result.iloc[0] == "direct"


def test_anaphora_resolved_stratum():
    df = _frame([_row(1, 0, mentions_target=True, resolved_by_anaphora=True)])
    result = goldset.assign_stratum(df)
    assert result.iloc[0] == "anaphora_resolved"


def test_ceo_only_stratum():
    df = _frame([_row(1, 0, mentions_ceo=True, mentions_target=False)])
    result = goldset.assign_stratum(df)
    assert result.iloc[0] == "ceo_only"


def test_other_only_stratum():
    df = _frame([_row(1, 0, mentions_other=True, mentions_target=False)])
    result = goldset.assign_stratum(df)
    assert result.iloc[0] == "other_only"


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------


def _build_corpus(n_direct=20, n_per_article=1):
    """A synthetic corpus with plenty of "direct" rows spread across many
    articles, for determinism / MAX_PER_ARTICLE tests."""
    rows = []
    article_id = 0
    sent_idx = 0
    for i in range(n_direct):
        rows.append(
            _row(article_id, sent_idx, text=f"Direct sentence {i}.", mentions_target=True)
        )
        sent_idx += 1
        if (i + 1) % n_per_article == 0:
            article_id += 1
            sent_idx = 0
    return _frame(rows)


def test_stratified_sample_is_deterministic_for_a_fixed_seed():
    df = _build_corpus(n_direct=30, n_per_article=1)
    sample1 = goldset.stratified_sample(df, n_per_stratum=10, seed=123)
    sample2 = goldset.stratified_sample(df, n_per_stratum=10, seed=123)
    assert set(sample1["sample_id"]) == set(sample2["sample_id"])


def test_stratified_sample_respects_max_per_article():
    # 5 articles, each with 6 "direct" sentences -- plenty of rows, but each
    # article may only contribute MAX_PER_ARTICLE to the sample.
    rows = []
    for article_id in range(5):
        for sent_idx in range(6):
            rows.append(
                _row(
                    article_id,
                    sent_idx,
                    text=f"Direct sentence {article_id}-{sent_idx}.",
                    mentions_target=True,
                )
            )
    df = _frame(rows)

    sample = goldset.stratified_sample(df, n_per_stratum=50, seed=1)
    direct_sample = sample[sample["stratum"] == "direct"]
    counts = direct_sample.groupby("article_id").size()
    assert (counts <= goldset.MAX_PER_ARTICLE).all()


def test_stratum_with_fewer_rows_than_requested_returns_all_and_does_not_raise():
    # Only 3 eligible "ceo_only" rows, across 3 distinct articles (so
    # MAX_PER_ARTICLE cannot itself be why fewer than requested come back).
    rows = [
        _row(article_id, 0, mentions_ceo=True, mentions_target=False)
        for article_id in range(3)
    ]
    df = _frame(rows)

    sample = goldset.stratified_sample(df, n_per_stratum=50, seed=1)
    ceo_sample = sample[sample["stratum"] == "ceo_only"]
    assert len(ceo_sample) == 3


# ---------------------------------------------------------------------------
# write_annotation_sheet
# ---------------------------------------------------------------------------


def test_write_annotation_sheet_withholds_model_outputs_and_leaves_blank_fields(tmp_path):
    df = _build_corpus(n_direct=5, n_per_article=1)
    sample = goldset.stratified_sample(df, n_per_stratum=5, seed=7)
    # Attach model-output-looking columns to prove they get withheld.
    sample["pos"] = 0.9
    sample["neg"] = 0.05
    sample["neu"] = 0.05
    sample["absa_pos"] = 0.8
    sample["absa_text"] = "Tesla delivered record numbers."
    sample["mentions_target"] = True

    out_path = tmp_path / "sheet.csv"
    goldset.write_annotation_sheet(sample, out_path)

    written = pd.read_csv(out_path, keep_default_na=False)
    assert list(written.columns) == [
        "sample_id",
        "stratum",
        "absa_aspect",
        "text",
        "label",
        "confidence",
        "notes",
    ]
    for withheld in ("pos", "neg", "neu", "absa_pos", "absa_text", "mentions_target"):
        assert withheld not in written.columns

    assert (written["label"] == "").all()
    assert (written["confidence"] == "").all()
    assert (written["notes"] == "").all()


# ---------------------------------------------------------------------------
# write_predictions
# ---------------------------------------------------------------------------


def test_write_predictions_joins_back_to_sheet_one_to_one(tmp_path):
    df = _build_corpus(n_direct=5, n_per_article=1)
    sample = goldset.stratified_sample(df, n_per_stratum=5, seed=7)
    sample["pos"] = 0.7
    sample["neg"] = 0.1
    sample["neu"] = 0.2

    sheet_path = tmp_path / "sheet.csv"
    predictions_path = tmp_path / "predictions.csv"
    goldset.write_annotation_sheet(sample, sheet_path)
    goldset.write_predictions(sample, predictions_path)

    sheet = pd.read_csv(sheet_path)
    predictions = pd.read_csv(predictions_path)

    assert len(predictions) == len(sample)
    assert "sample_id" in predictions.columns

    joined = sheet.merge(predictions, on="sample_id", how="inner", validate="one_to_one")
    assert len(joined) == len(sheet)


def test_write_predictions_does_not_raise_when_absa_columns_absent(tmp_path):
    # A sentence table that predates ABSA/coref entirely.
    rows = [
        {
            "article_id": 1,
            "sent_idx": 0,
            "text": "Tesla delivered record numbers.",
            "mentions_target": True,
            "mentions_other": False,
            "mentions_ceo": False,
            "is_comparative": False,
            "is_boilerplate": False,
        }
    ]
    df = _frame(rows)
    stratum = goldset.assign_stratum(df)
    df["stratum"] = stratum
    df["sample_id"] = "direct-1-0"

    out_path = tmp_path / "predictions.csv"
    goldset.write_predictions(df, out_path)

    written = pd.read_csv(out_path)
    assert list(written["sample_id"]) == ["direct-1-0"]
    assert "absa_pos" not in written.columns
    assert "resolved_by_coref" not in written.columns
