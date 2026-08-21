import numpy as np
import pandas as pd
import pytest

from stock_predictor.config import LEAD_SENTENCE_WINDOW
from stock_predictor.text.fusion import (
    AGGREGATED_VARIANTS,
    CONF_FLOOR,
    DEADBAND,
    FUSION_AGGREGATIONS,
    PROVENANCE_CHANNELS,
    VARIANTS,
    aggregate_fusion_features,
    aggregate_provenance_features,
    provenance_channel,
    score_variants,
    signed,
)


def _frame(rows, index=None):
    df = pd.DataFrame(rows)
    if index is not None:
        df.index = index
    return df


# ---------------------------------------------------------------------------
# signed()
# ---------------------------------------------------------------------------


def test_signed_is_pos_minus_neg():
    pos = pd.Series([0.9, 0.1, 0.5])
    neg = pd.Series([0.1, 0.8, 0.5])
    result = signed(pos, neg)
    assert result.tolist() == pytest.approx([0.8, -0.7, 0.0])


def test_signed_is_nan_safe():
    pos = pd.Series([0.9, np.nan])
    neg = pd.Series([np.nan, 0.2])
    result = signed(pos, neg)
    assert result.isna().tolist() == [True, True]


# ---------------------------------------------------------------------------
# score_variants -- hand-computed expected values for every variant
# ---------------------------------------------------------------------------


def test_fin_is_signed_finbert():
    df = _frame([{"pos": 0.7, "neg": 0.2, "neu": 0.1, "absa_pos": 0.5, "absa_neg": 0.5, "absa_neu": 0.0}])
    out = score_variants(df)
    assert out["fin"].iloc[0] == pytest.approx(0.5)


def test_absa_is_signed_absa():
    df = _frame([{"pos": 0.5, "neg": 0.5, "neu": 0.0, "absa_pos": 0.8, "absa_neg": 0.1, "absa_neu": 0.1}])
    out = score_variants(df)
    assert out["absa"].iloc[0] == pytest.approx(0.7)


def test_mean_blend_is_plain_average_of_fin_and_absa():
    # fin = 0.9 - 0.1 = 0.8, absa = 0.2 - 0.6 = -0.4 -> mean_blend = 0.2
    df = _frame(
        [{"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.2, "absa_neg": 0.6, "absa_neu": 0.2}]
    )
    out = score_variants(df)
    assert out["mean_blend"].iloc[0] == pytest.approx(0.2)


def test_gated_downweights_by_absa_neu_relative_to_low_neu_same_fin():
    # Same fin (0.8) on both rows; only absa_neu differs. High absa_neu
    # should shrink the gated score toward zero relative to low absa_neu.
    df = _frame(
        [
            {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.4, "absa_neg": 0.4, "absa_neu": 0.9},
            {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.4, "absa_neg": 0.4, "absa_neu": 0.1},
        ]
    )
    out = score_variants(df)
    high_neu_gated = out["gated"].iloc[0]
    low_neu_gated = out["gated"].iloc[1]
    assert high_neu_gated == pytest.approx(0.8 * (1 - 0.9))
    assert low_neu_gated == pytest.approx(0.8 * (1 - 0.1))
    assert abs(high_neu_gated) < abs(low_neu_gated)


def test_agree_only_returns_fin_on_agreement_and_zero_not_nan_on_disagreement():
    df = _frame(
        [
            # agree: both positive
            {"pos": 0.8, "neg": 0.2, "neu": 0.0, "absa_pos": 0.6, "absa_neg": 0.3, "absa_neu": 0.1},
            # disagree: fin positive, absa negative
            {"pos": 0.8, "neg": 0.2, "neu": 0.0, "absa_pos": 0.2, "absa_neg": 0.7, "absa_neu": 0.1},
            # (0, 0) counts as agreement
            {"pos": 0.5, "neg": 0.5, "neu": 0.0, "absa_pos": 0.4, "absa_neg": 0.4, "absa_neu": 0.2},
        ]
    )
    out = score_variants(df)
    assert out["agree_only"].iloc[0] == pytest.approx(0.6)  # fin on agreement
    assert out["agree_only"].iloc[1] == 0.0  # exact zero, not NaN, on disagreement
    assert not pd.isna(out["agree_only"].iloc[1])
    assert out["agree_only"].iloc[2] == pytest.approx(0.0)  # (0, 0) treated as agreement -> fin (=0)


# ---------------------------------------------------------------------------
# sign_graft -- the double-negative bug this variant exists to avoid
# ---------------------------------------------------------------------------


def test_sign_graft_is_a_graft_not_a_product_both_models_negative_stays_negative():
    # fin = 0.05 - 0.95 = -0.9, absa = 0.05 - 0.95 = -0.9.
    # A naive product fin * absa = (-0.9) * (-0.9) = +0.81 (WRONG: silently
    # flips two agreeing-negative scores into a positive result). The graft
    # must instead report a NEGATIVE score, since both models agree the
    # sentence is negative and ABSA's sign should be grafted onto FinBERT's
    # magnitude, not multiplied against it.
    df = _frame(
        [{"pos": 0.05, "neg": 0.95, "neu": 0.0, "absa_pos": 0.05, "absa_neg": 0.95, "absa_neu": 0.0}]
    )
    out = score_variants(df)
    naive_product = (0.05 - 0.95) * (0.05 - 0.95)
    assert naive_product == pytest.approx(0.81)  # sanity: the bug this guards against
    assert out["sign_graft"].iloc[0] < 0
    assert out["sign_graft"].iloc[0] == pytest.approx(-0.9)  # sign(absa) * |fin| = -1 * 0.9


def test_sign_graft_deadband_boundary():
    # The module default DEADBAND is now 0.0 (see fusion.py's comment on
    # DEADBAND for the measured evidence), so this test exercises the
    # boundary behaviour by passing an EXPLICIT non-zero deadband instead of
    # relying on the module-level constant.
    explicit_deadband = 0.10
    fin_pos, fin_neg = 0.9, 0.1  # fin = 0.8
    just_below = explicit_deadband - 0.001
    just_above = explicit_deadband + 0.001
    # absa signed score just below/above the deadband, built from pos/neg
    # such that absa = pos - neg equals the target value exactly, with
    # absa_neu filled in so rows sum sensibly (not required to sum to 1 for
    # this module, but keep it plausible).
    df = _frame(
        [
            {
                "pos": fin_pos,
                "neg": fin_neg,
                "neu": 0.0,
                "absa_pos": 0.5 + just_below / 2,
                "absa_neg": 0.5 - just_below / 2,
                "absa_neu": 0.0,
            },
            {
                "pos": fin_pos,
                "neg": fin_neg,
                "neu": 0.0,
                "absa_pos": 0.5 + just_above / 2,
                "absa_neg": 0.5 - just_above / 2,
                "absa_neu": 0.0,
            },
        ]
    )
    out = score_variants(df, deadband=explicit_deadband)
    assert out["sign_graft"].iloc[0] == 0.0  # |absa| just below deadband -> exactly 0.0
    assert out["sign_graft"].iloc[1] == pytest.approx(0.8)  # |absa| just above -> full grafted magnitude


def test_sign_graft_default_deadband_is_zero_and_keeps_small_absa_signs():
    """Regression pinning the measured sweep in fusion.py's DEADBAND comment:
    corpus-weighted directional accuracy of sign_graft is monotonically
    HARMFUL as the deadband grows (0.918 at deadband=0.0 down to 0.431 at
    deadband=0.20), because ABSA's sign is informative even at small
    magnitude -- 17 of 17 hand-checked small-|absa| directional sentences
    had ABSA's sign correct. With the default deadband (0.0), a row with a
    small-magnitude ABSA score must still produce a non-zero graft carrying
    ABSA's sign, not be zeroed out.
    """
    # fin = 0.1 - 0.9 = -0.8, absa = 0.515 - 0.485 = +0.03 (small magnitude,
    # positive sign). Under the old DEADBAND=0.10 default this row would
    # have been zeroed; under the evidence-backed default of 0.0 it must
    # instead carry ABSA's sign onto FinBERT's magnitude: +1 * |-0.8| = +0.8.
    df = _frame(
        [{"pos": 0.1, "neg": 0.9, "neu": 0.0, "absa_pos": 0.515, "absa_neg": 0.485, "absa_neu": 0.0}]
    )
    out = score_variants(df)  # no explicit deadband -> uses the default (0.0)
    assert out["sign_graft"].iloc[0] == pytest.approx(0.8)  # sign(+0.03) * |-0.8| = +0.8, not zeroed


# ---------------------------------------------------------------------------
# conf_graft / conf_graft_soft -- confidence-weighted grafts
# ---------------------------------------------------------------------------


def test_conf_graft_hand_computed():
    # fin = 0.9 - 0.1 = 0.8, absa = 0.7 - 0.2 = 0.5 -> conf_graft = absa * |fin| = 0.5 * 0.8 = 0.4
    df = _frame(
        [{"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.7, "absa_neg": 0.2, "absa_neu": 0.1}]
    )
    out = score_variants(df)
    assert out["conf_graft"].iloc[0] == pytest.approx(0.4)


def test_conf_graft_soft_hand_computed():
    # fin = 0.9 - 0.1 = 0.8, absa = 0.7 - 0.2 = 0.5.
    # conf_graft_soft = sign(absa) * |fin| * (0.5 + 0.5 * |absa|)
    #                 = 1 * 0.8 * (0.5 + 0.25) = 0.8 * 0.75 = 0.6
    df = _frame(
        [{"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.7, "absa_neg": 0.2, "absa_neu": 0.1}]
    )
    out = score_variants(df)
    assert out["conf_graft_soft"].iloc[0] == pytest.approx(0.6)


def test_conf_graft_floor_hand_computed():
    # fin = 0.9 - 0.1 = 0.8, absa = 0.7 - 0.2 = 0.5.
    # conf_graft_floor = sign(absa) * |fin| * (CONF_FLOOR + (1 - CONF_FLOOR) * |absa|)
    #                  = 1 * 0.8 * (0.7 + 0.3 * 0.5) = 0.8 * 0.85 = 0.68
    df = _frame(
        [{"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.7, "absa_neg": 0.2, "absa_neu": 0.1}]
    )
    out = score_variants(df)
    assert out["conf_graft_floor"].iloc[0] == pytest.approx(0.68)


def test_conf_floor_is_the_shipped_value():
    # The measured table in CONF_FLOOR's comment is what justifies 0.7. If
    # this value is changed, that table no longer describes what ships --
    # this test exists so the change cannot pass silently.
    assert CONF_FLOOR == 0.7


def test_conf_graft_floor_interpolates_conf_graft_and_sign_graft():
    # The whole family is one formula in one parameter. At floor 0 it must
    # reproduce conf_graft exactly; at floor 1, sign_graft. Pinning the two
    # endpoints pins the parameterisation itself, so a future floor change
    # cannot quietly become a different formula.
    rows = [
        {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.7, "absa_neg": 0.2, "absa_neu": 0.1},
        {"pos": 0.2, "neg": 0.6, "neu": 0.2, "absa_pos": 0.1, "absa_neg": 0.8, "absa_neu": 0.1},
        {"pos": 0.4, "neg": 0.4, "neu": 0.2, "absa_pos": 0.5, "absa_neg": 0.1, "absa_neu": 0.4},
    ]
    out = score_variants(_frame(rows))
    fin = out["fin"].abs()
    absa = out["absa"]
    at_zero = np.sign(absa) * fin * (0.0 + 1.0 * absa.abs())
    at_one = np.sign(absa) * fin * (1.0 + 0.0 * absa.abs())
    pd.testing.assert_series_equal(at_zero, out["conf_graft"], check_names=False)
    pd.testing.assert_series_equal(at_one, out["sign_graft"], check_names=False)


def test_conf_graft_floor_never_shrinks_magnitude_below_conf_graft():
    # The floor exists to stop a hesitant ABSA call crushing a confident
    # FinBERT magnitude. Since 0 <= |absa| <= 1 and CONF_FLOOR > 0, the
    # floored variant must be at least as loud as conf_graft on EVERY row,
    # and strictly louder wherever |absa| < 1. This is the property the
    # 2,500-row audit was measuring; pin it so it cannot regress.
    rows = [
        {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.7, "absa_neg": 0.2, "absa_neu": 0.1},
        {"pos": 0.2, "neg": 0.6, "neu": 0.2, "absa_pos": 0.1, "absa_neg": 0.8, "absa_neu": 0.1},
        {"pos": 0.5, "neg": 0.4, "neu": 0.1, "absa_pos": 0.30, "absa_neg": 0.28, "absa_neu": 0.42},
    ]
    out = score_variants(_frame(rows))
    assert (out["conf_graft_floor"].abs() >= out["conf_graft"].abs() - 1e-12).all()
    hesitant = out["absa"].abs() < 1.0
    assert (
        out.loc[hesitant, "conf_graft_floor"].abs() > out.loc[hesitant, "conf_graft"].abs()
    ).all()


def test_conf_graft_floor_preserves_absa_sign_and_propagates_nan():
    # Same two guarantees the other grafts carry. The sign guarantee is what
    # makes the floor safe to raise at all -- no floor value can flip a row's
    # direction, so the audit's referent verdicts stay valid across floors.
    # NaN: np.sign(np.nan) is nan, so the product poisons with no .where()
    # guard, exactly as for conf_graft_soft.
    rows = [
        {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.1, "absa_neg": 0.8, "absa_neu": 0.1},
        {"pos": 0.1, "neg": 0.8, "neu": 0.1, "absa_pos": 0.7, "absa_neg": 0.2, "absa_neu": 0.1},
        {"pos": np.nan, "neg": 0.1, "neu": 0.0, "absa_pos": 0.7, "absa_neg": 0.2, "absa_neu": 0.1},
        {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": np.nan, "absa_neg": 0.2, "absa_neu": 0.1},
    ]
    out = score_variants(_frame(rows))
    # Row 0: FinBERT positive, ABSA negative -> ABSA's sign must win.
    assert out["conf_graft_floor"].iloc[0] < 0
    # Row 1: FinBERT negative, ABSA positive -> ABSA's sign must win.
    assert out["conf_graft_floor"].iloc[1] > 0
    assert np.isnan(out["conf_graft_floor"].iloc[2])
    assert np.isnan(out["conf_graft_floor"].iloc[3])


def test_only_the_shipped_variant_is_aggregated_to_article_level():
    """The earlier variants stay computable per sentence so old measurements
    remain reproducible, but they are not carried into the feature table: they
    are the same formula at a different floor and correlate at r ~= 0.9, so
    promoting all three spent twelve columns restating one number."""
    assert AGGREGATED_VARIANTS == ("conf_graft_floor",)
    for kept in ("conf_graft", "conf_graft_soft", "sign_graft"):
        assert kept in VARIANTS, f"{kept} must remain available per sentence"
        assert kept not in AGGREGATED_VARIANTS


def test_conf_graft_equals_sign_graft_scaled_by_absa_confidence():
    # Pins the algebraic identity documented in fusion.py: conf_graft ==
    # sign(absa) * |fin| * |absa| == sign_graft * |absa|, since
    # sign(x) * |x| == x for any real x.
    df = _frame(
        [
            {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.7, "absa_neg": 0.2, "absa_neu": 0.1},
            {"pos": 0.1, "neg": 0.9, "neu": 0.0, "absa_pos": 0.3, "absa_neg": 0.6, "absa_neu": 0.1},
            {"pos": 0.5, "neg": 0.5, "neu": 0.0, "absa_pos": 0.5, "absa_neg": 0.5, "absa_neu": 0.0},
        ]
    )
    out = score_variants(df, deadband=0.0)
    expected = out["sign_graft"] * out["absa"].abs()
    assert out["conf_graft"].tolist() == pytest.approx(expected.tolist())


def test_conf_graft_hesitant_vs_confident_absa_diverge_while_sign_graft_does_not():
    # Same fin on both rows, absa differs only in magnitude (0.05 vs 0.95),
    # both positive. sign_graft discards ABSA's magnitude entirely, so it
    # gives both rows the IDENTICAL magnitude -- that is the pathology
    # conf_graft exists to fix. conf_graft must instead produce very
    # different magnitudes for the hesitant vs. confident ABSA call.
    df = _frame(
        [
            # hesitant: absa = 0.525 - 0.475 = 0.05
            {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.525, "absa_neg": 0.475, "absa_neu": 0.0},
            # confident: absa = 0.975 - 0.025 = 0.95
            {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.975, "absa_neg": 0.025, "absa_neu": 0.0},
        ]
    )
    out = score_variants(df)
    hesitant_sign_graft = out["sign_graft"].iloc[0]
    confident_sign_graft = out["sign_graft"].iloc[1]
    assert hesitant_sign_graft == pytest.approx(confident_sign_graft)  # sign_graft: identical

    hesitant_conf_graft = out["conf_graft"].iloc[0]
    confident_conf_graft = out["conf_graft"].iloc[1]
    assert hesitant_conf_graft == pytest.approx(0.05 * 0.8)
    assert confident_conf_graft == pytest.approx(0.95 * 0.8)
    assert abs(hesitant_conf_graft) < abs(confident_conf_graft)  # conf_graft: diverges


def test_conf_graft_soft_multiplier_stays_within_half_to_one_range():
    # absa near 0 -> multiplier near 0.5, NOT crushed to 0: the row must
    # still retain half of |fin|.
    df = _frame(
        [
            # absa = 0.001 (essentially zero magnitude)
            {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.5005, "absa_neg": 0.4995, "absa_neu": 0.0},
        ]
    )
    out = score_variants(df)
    # multiplier ~= 0.5 -> conf_graft_soft ~= sign(absa) * 0.8 * 0.5 = 0.4
    assert out["conf_graft_soft"].iloc[0] == pytest.approx(0.8 * 0.5, abs=1e-3)
    assert abs(out["conf_graft_soft"].iloc[0]) >= 0.8 * 0.5 - 1e-3  # never below the 0.5 floor


def test_conf_graft_and_conf_graft_soft_preserve_absa_sign():
    df = _frame(
        [
            {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.8, "absa_neg": 0.1, "absa_neu": 0.1},
            {"pos": 0.9, "neg": 0.1, "neu": 0.0, "absa_pos": 0.1, "absa_neg": 0.8, "absa_neu": 0.1},
            {"pos": 0.1, "neg": 0.9, "neu": 0.0, "absa_pos": 0.8, "absa_neg": 0.1, "absa_neu": 0.1},
            {"pos": 0.1, "neg": 0.9, "neu": 0.0, "absa_pos": 0.1, "absa_neg": 0.8, "absa_neu": 0.1},
        ]
    )
    out = score_variants(df)
    for col in ("conf_graft", "conf_graft_soft"):
        for absa_val, score in zip(out["absa"], out[col]):
            if absa_val != 0:
                assert np.sign(score) == np.sign(absa_val), f"{col} sign mismatch for absa={absa_val}"


# ---------------------------------------------------------------------------
# NaN handling -- 0.0 and NaN must never be conflated
# ---------------------------------------------------------------------------


def test_nan_in_finbert_gives_nan_for_every_variant_reading_it():
    df = _frame(
        [{"pos": np.nan, "neg": np.nan, "neu": np.nan, "absa_pos": 0.6, "absa_neg": 0.3, "absa_neu": 0.1}]
    )
    out = score_variants(df)
    row = out.iloc[0]
    assert row["fin"] is np.nan or pd.isna(row["fin"])
    # absa does not read FinBERT's score, so it alone stays a real number.
    assert not pd.isna(row["absa"])
    for col in ("fin", "mean_blend", "sign_graft", "conf_graft", "conf_graft_soft", "gated", "agree_only"):
        assert pd.isna(row[col]), f"{col} should be NaN when FinBERT score is missing, got {row[col]!r}"


def test_nan_in_absa_gives_nan_for_every_variant_reading_it():
    df = _frame(
        [{"pos": 0.7, "neg": 0.2, "neu": 0.1, "absa_pos": np.nan, "absa_neg": np.nan, "absa_neu": np.nan}]
    )
    out = score_variants(df)
    row = out.iloc[0]
    # fin does not read absa, so it must still be a real number.
    assert not pd.isna(row["fin"])
    for col in ("absa", "mean_blend", "sign_graft", "conf_graft", "conf_graft_soft", "gated", "agree_only"):
        assert pd.isna(row[col]), f"{col} should be NaN when ABSA score is missing, got {row[col]!r}"


# ---------------------------------------------------------------------------
# Structural: columns, index alignment
# ---------------------------------------------------------------------------


def test_output_columns_match_variants_exactly():
    df = _frame(
        [{"pos": 0.7, "neg": 0.2, "neu": 0.1, "absa_pos": 0.6, "absa_neg": 0.3, "absa_neu": 0.1}]
    )
    out = score_variants(df)
    assert list(out.columns) == list(VARIANTS)


def test_output_is_index_aligned_to_input_including_non_contiguous_index():
    rows = [
        {"pos": 0.7, "neg": 0.2, "neu": 0.1, "absa_pos": 0.6, "absa_neg": 0.3, "absa_neu": 0.1},
        {"pos": 0.3, "neg": 0.6, "neu": 0.1, "absa_pos": 0.2, "absa_neg": 0.7, "absa_neu": 0.1},
        {"pos": 0.5, "neg": 0.5, "neu": 0.0, "absa_pos": 0.5, "absa_neg": 0.5, "absa_neu": 0.0},
    ]
    df = _frame(rows, index=[10, 42, 7])
    out = score_variants(df)
    assert list(out.index) == [10, 42, 7]

    # Also check a filtered (non-contiguous, out-of-order) subset preserves
    # its exact index rather than resetting or sorting it.
    subset = df.loc[[42, 10]]
    out_subset = score_variants(subset)
    assert list(out_subset.index) == [42, 10]


# ---------------------------------------------------------------------------
# aggregate_fusion_features -- article-level conf_graft / conf_graft_soft
# ---------------------------------------------------------------------------


def _fus_row(
    article_id,
    sent_idx,
    mentions_target,
    pos,
    neg,
    absa_pos,
    absa_neg,
    absa_neu=0.1,
    is_boilerplate=False,
):
    return {
        "article_id": article_id,
        "sent_idx": sent_idx,
        "mentions_target": mentions_target,
        "is_boilerplate": is_boilerplate,
        "pos": pos,
        "neg": neg,
        "neu": max(0.0, 1 - pos - neg),
        "absa_pos": absa_pos,
        "absa_neg": absa_neg,
        "absa_neu": absa_neu,
    }


def test_top3_neg_is_not_diluted_by_neutral_filler():
    """This is the failure mode fus_<variant>_top3_neg exists to fix: a plain
    mean is diluted by filler sentences, so it can rank a short, blandly
    mildly-negative article as MORE negative than a longer article that
    actually carries a few strongly negative target sentences buried in a lot
    of neutral/mildly-positive filler. _top3_neg reads the tail directly and
    is not fooled by the filler; _mean is.
    """
    # Article A: two mildly negative target sentences, nothing else. Both
    # models agree, mildly negative -> conf_graft is a small negative number
    # for each.
    article_a = [
        _fus_row("A", 0, True, pos=0.35, neg=0.45, absa_pos=0.4, absa_neg=0.5),
        _fus_row("A", 1, True, pos=0.35, neg=0.45, absa_pos=0.4, absa_neg=0.5),
    ]
    # Article B: three STRONGLY negative target sentences, plus a lot of
    # mildly-positive target filler. There is enough filler (15 sentences) to
    # pull the plain mean above zero despite the strongly negative tail --
    # that dilution is exactly the failure mode this test demonstrates.
    strong_negative = [
        _fus_row("B", 0, True, pos=0.02, neg=0.97, absa_pos=0.02, absa_neg=0.97),
        _fus_row("B", 1, True, pos=0.02, neg=0.97, absa_pos=0.02, absa_neg=0.97),
        _fus_row("B", 2, True, pos=0.03, neg=0.96, absa_pos=0.03, absa_neg=0.96),
    ]
    filler = [
        _fus_row("B", 3 + i, True, pos=0.55, neg=0.1, absa_pos=0.55, absa_neg=0.1) for i in range(15)
    ]
    article_b = strong_negative + filler
    df = pd.DataFrame(article_a + article_b)
    result = aggregate_fusion_features(df).set_index("article_id")

    # The dilution problem, demonstrated: B's mean is LESS negative (or even
    # positive) than A's mean, even though B contains the stronger negative
    # message.
    assert result.loc["B", "fus_conf_graft_floor_mean"] > result.loc["A", "fus_conf_graft_floor_mean"]

    # _top3_neg correctly ranks B as more negative than A -- it reads B's
    # strongly-negative tail instead of being diluted by B's filler.
    assert result.loc["B", "fus_conf_graft_floor_top3_neg"] < result.loc["A", "fus_conf_graft_floor_top3_neg"]


def test_fewer_than_three_population_sentences_averages_what_exists():
    df = pd.DataFrame(
        [
            _fus_row("C", 0, True, pos=0.8, neg=0.1, absa_pos=0.7, absa_neg=0.2),
            _fus_row("C", 1, True, pos=0.2, neg=0.7, absa_pos=0.3, absa_neg=0.6),
        ]
    )
    result = aggregate_fusion_features(df).set_index("article_id")
    row = result.loc["C"]

    variants = score_variants(df)
    expected_mean = variants["conf_graft_floor"].mean()
    assert row["fus_conf_graft_floor_mean"] == pytest.approx(expected_mean)
    # Only 2 rows exist -- top3_pos/top3_neg average both of them (not padded,
    # not NaN just because there weren't 3).
    assert row["fus_conf_graft_floor_top3_pos"] == pytest.approx(expected_mean)
    assert row["fus_conf_graft_floor_top3_neg"] == pytest.approx(expected_mean)


def test_empty_population_gives_nan_not_zero_for_all_four_stats():
    # No mentions_target rows at all for this article.
    df = pd.DataFrame(
        [
            _fus_row("D", 0, False, pos=0.5, neg=0.3, absa_pos=0.4, absa_neg=0.3),
        ]
    )
    result = aggregate_fusion_features(df).set_index("article_id")
    row = result.loc["D"]
    for variant in AGGREGATED_VARIANTS:
        for agg in FUSION_AGGREGATIONS:
            val = row[f"fus_{variant}_{agg}"]
            if agg == "lead":
                # Documented exception: an article that never mentions the target
                # early delivered no early sentiment, which is a measurement.
                assert val == 0.0, f"fus_{variant}_lead should be 0.0, got {val!r}"
            else:
                assert pd.isna(val), f"fus_{variant}_{agg} should be NaN, got {val!r}"


def test_top3_pos_on_all_negative_article_returns_mean_of_least_negative_not_nan():
    df = pd.DataFrame(
        [
            _fus_row("E", 0, True, pos=0.05, neg=0.9, absa_pos=0.05, absa_neg=0.9),
            _fus_row("E", 1, True, pos=0.1, neg=0.8, absa_pos=0.1, absa_neg=0.8),
            _fus_row("E", 2, True, pos=0.2, neg=0.6, absa_pos=0.2, absa_neg=0.6),
            _fus_row("E", 3, True, pos=0.02, neg=0.95, absa_pos=0.02, absa_neg=0.95),
        ]
    )
    result = aggregate_fusion_features(df).set_index("article_id")
    row = result.loc["E"]

    variants = score_variants(df)
    conf_graft = variants["conf_graft_floor"]
    expected_top3_pos = conf_graft.nlargest(3).mean()

    assert not pd.isna(row["fus_conf_graft_floor_top3_pos"])
    assert row["fus_conf_graft_floor_top3_pos"] == pytest.approx(expected_top3_pos)
    assert row["fus_conf_graft_floor_top3_pos"] < 0  # still negative -- article IS all-negative
    # It is the LEAST-negative of the four, i.e. greater than the plain mean
    # (closer to zero), not equal to the most-negative single value.
    assert row["fus_conf_graft_floor_top3_pos"] > conf_graft.min()


def test_lead_respects_lead_sentence_window_and_ignores_later_sentences():
    # One target sentence inside the lead window, one just after it, with
    # opposite-sign scores so the two are easy to tell apart.
    df = pd.DataFrame(
        [
            _fus_row("F", 0, True, pos=0.9, neg=0.05, absa_pos=0.8, absa_neg=0.1),
            _fus_row(
                "F", LEAD_SENTENCE_WINDOW, True, pos=0.05, neg=0.9, absa_pos=0.1, absa_neg=0.8
            ),
        ]
    )
    result = aggregate_fusion_features(df).set_index("article_id")
    row = result.loc["F"]

    variants = score_variants(df)
    lead_only_expected = variants["conf_graft_floor"].iloc[0]  # only the sent_idx=0 row
    assert row["fus_conf_graft_floor_lead"] == pytest.approx(lead_only_expected)
    assert row["fus_conf_graft_floor_lead"] > 0  # positive lead sentence, not diluted by the later negative one


def test_lead_is_zero_not_nan_when_no_lead_window_target_sentences():
    """The one deliberate exception to NaN-over-zero. _lead measures how much
    sentiment an article delivers EARLY, so an article that never mentions the
    target in its opening window delivered none, and 0.0 is that measurement
    rather than a missing one. See aggregate_fusion_features()."""
    df = pd.DataFrame(
        [
            _fus_row(
                "G", LEAD_SENTENCE_WINDOW, True, pos=0.7, neg=0.2, absa_pos=0.6, absa_neg=0.3
            ),
        ]
    )
    result = aggregate_fusion_features(df).set_index("article_id")
    assert result.loc["G", "fus_conf_graft_floor_lead"] == 0.0
    # The other aggregations, which have no such exception, stay NaN-free here
    # because the article does have a target sentence, just not an early one.
    assert not pd.isna(result.loc["G", "fus_conf_graft_floor_mean"])


def test_boilerplate_and_non_target_sentences_excluded_from_population():
    df = pd.DataFrame(
        [
            _fus_row("H", 0, True, pos=0.8, neg=0.1, absa_pos=0.7, absa_neg=0.2),
            # Boilerplate target sentence -- must be excluded despite mentions_target.
            _fus_row(
                "H", 1, True, pos=0.05, neg=0.95, absa_pos=0.05, absa_neg=0.95, is_boilerplate=True
            ),
            # Non-target sentence -- must be excluded despite a real score.
            _fus_row("H", 2, False, pos=0.05, neg=0.95, absa_pos=0.05, absa_neg=0.95),
        ]
    )
    result = aggregate_fusion_features(df).set_index("article_id")
    row = result.loc["H"]

    only_row = df.iloc[[0]]
    expected = score_variants(only_row)["conf_graft_floor"].iloc[0]
    assert row["fus_conf_graft_floor_mean"] == pytest.approx(expected)
    assert row["fus_conf_graft_floor_top3_pos"] == pytest.approx(expected)
    assert row["fus_conf_graft_floor_top3_neg"] == pytest.approx(expected)


def test_median_is_robust_to_a_single_extreme_sentence():
    """This is the 21.5%-of-articles case _median exists to expose (see
    aggregate_fusion_features()'s docstring: on this corpus 206 of 958
    articles with >= 5 target sentences have a mean and median that disagree
    on SIGN). Here four mildly-positive target sentences are joined by one
    very negative outlier: the outlier is enough to drag the MEAN negative,
    but the MEDIAN -- reading the robust centre, not the extremes -- stays
    positive. That divergence is exactly the skew signal _median is for.
    """
    df = pd.DataFrame(
        [
            _fus_row("I", 0, True, pos=0.6, neg=0.3, absa_pos=0.6, absa_neg=0.3),
            _fus_row("I", 1, True, pos=0.6, neg=0.3, absa_pos=0.6, absa_neg=0.3),
            _fus_row("I", 2, True, pos=0.6, neg=0.3, absa_pos=0.6, absa_neg=0.3),
            _fus_row("I", 3, True, pos=0.6, neg=0.3, absa_pos=0.6, absa_neg=0.3),
            # One very negative outlier, large enough to pull the mean across
            # zero despite four mildly-positive sentences.
            _fus_row("I", 4, True, pos=0.02, neg=0.99, absa_pos=0.02, absa_neg=0.99),
        ]
    )
    result = aggregate_fusion_features(df).set_index("article_id")
    row = result.loc["I"]

    variants = score_variants(df)
    conf_graft = variants["conf_graft_floor"]
    assert row["fus_conf_graft_floor_mean"] == pytest.approx(conf_graft.mean())
    assert row["fus_conf_graft_floor_median"] == pytest.approx(conf_graft.median())

    # The point: mean and median disagree on SIGN.
    assert row["fus_conf_graft_floor_mean"] < 0
    assert row["fus_conf_graft_floor_median"] > 0


def test_spread_separates_contested_from_quiet_articles():
    """Two articles with near-identical (small) means but very different
    spreads -- the CONTESTED-vs-QUIET distinction _spread exists to expose
    (see aggregate_fusion_features()'s docstring). A plain mean cannot tell
    a article with loud claims in both directions from one where nothing
    strong was said at all; _spread (top3_pos - top3_neg) can.
    """
    # Contested: loud positive AND loud negative target sentences, roughly
    # cancelling in the mean.
    contested = [
        _fus_row("J", 0, True, pos=0.95, neg=0.03, absa_pos=0.95, absa_neg=0.03),
        _fus_row("J", 1, True, pos=0.93, neg=0.05, absa_pos=0.93, absa_neg=0.05),
        _fus_row("J", 2, True, pos=0.02, neg=0.95, absa_pos=0.02, absa_neg=0.95),
        _fus_row("J", 3, True, pos=0.03, neg=0.94, absa_pos=0.03, absa_neg=0.94),
    ]
    # Quiet: only mild sentences, nothing loud in either direction, also
    # roughly cancelling to a near-zero mean.
    quiet = [
        _fus_row("K", 0, True, pos=0.55, neg=0.45, absa_pos=0.55, absa_neg=0.45),
        _fus_row("K", 1, True, pos=0.52, neg=0.48, absa_pos=0.52, absa_neg=0.48),
        _fus_row("K", 2, True, pos=0.45, neg=0.55, absa_pos=0.45, absa_neg=0.55),
        _fus_row("K", 3, True, pos=0.48, neg=0.52, absa_pos=0.48, absa_neg=0.52),
    ]
    df = pd.DataFrame(contested + quiet)
    result = aggregate_fusion_features(df).set_index("article_id")

    contested_row = result.loc["J"]
    quiet_row = result.loc["K"]

    # Means are close -- a mean-only view could not tell these apart.
    assert contested_row["fus_conf_graft_floor_mean"] == pytest.approx(
        quiet_row["fus_conf_graft_floor_mean"], abs=0.05
    )
    # Spreads are far apart -- spread makes the distinction the mean cannot.
    assert contested_row["fus_conf_graft_floor_spread"] > quiet_row["fus_conf_graft_floor_spread"] + 0.5


def test_spread_is_zero_for_single_sentence_population():
    df = pd.DataFrame(
        [
            _fus_row("L", 0, True, pos=0.7, neg=0.2, absa_pos=0.6, absa_neg=0.3),
        ]
    )
    result = aggregate_fusion_features(df).set_index("article_id")
    row = result.loc["L"]
    # A single-sentence population: top3_pos and top3_neg both equal that
    # one score, so their difference is exactly 0.0, not NaN.
    assert row["fus_conf_graft_floor_top3_pos"] == pytest.approx(row["fus_conf_graft_floor_top3_neg"])
    assert row["fus_conf_graft_floor_spread"] == pytest.approx(0.0)
    assert not pd.isna(row["fus_conf_graft_floor_spread"])


def test_spread_and_median_are_nan_on_empty_population():
    # No mentions_target rows at all for this article -- reuses the same
    # empty-population setup as test_empty_population_gives_nan_not_zero_for_all_four_stats.
    df = pd.DataFrame(
        [
            _fus_row("M", 0, False, pos=0.5, neg=0.3, absa_pos=0.4, absa_neg=0.3),
        ]
    )
    result = aggregate_fusion_features(df).set_index("article_id")
    row = result.loc["M"]
    for variant in AGGREGATED_VARIANTS:
        assert pd.isna(row[f"fus_{variant}_median"])
        assert pd.isna(row[f"fus_{variant}_spread"])


def test_empty_input_frame_returns_empty_frame_with_correct_columns():
    df = pd.DataFrame(
        columns=[
            "article_id",
            "sent_idx",
            "mentions_target",
            "is_boilerplate",
            "pos",
            "neg",
            "absa_pos",
            "absa_neg",
            "absa_neu",
        ]
    )
    result = aggregate_fusion_features(df)
    assert len(result) == 0
    expected_cols = ["article_id"] + [
        f"fus_{variant}_{agg}" for variant in AGGREGATED_VARIANTS for agg in FUSION_AGGREGATIONS
    ]
    assert list(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# provenance split -- HOW a sentence entered the target population
# ---------------------------------------------------------------------------


def _prov_row(
    article_id,
    sent_idx,
    channel,
    pos=0.6,
    neg=0.2,
    absa_pos=0.7,
    absa_neg=0.1,
    absa_neu=0.2,
    is_boilerplate=False,
    mentions_target=True,
):
    """A sentence row whose provenance flags put it in `channel`.

    channel is one of PROVENANCE_CHANNELS, or None for a non-target row.
    """
    row = _fus_row(
        article_id,
        sent_idx,
        mentions_target,
        pos=pos,
        neg=neg,
        absa_pos=absa_pos,
        absa_neg=absa_neg,
        absa_neu=absa_neu,
        is_boilerplate=is_boilerplate,
    )
    row["resolved_by_coref"] = channel in ("coref_span", "coref_nospan")
    row["mention_char_start"] = 3 if channel == "coref_span" else None
    row["mention_char_end"] = 8 if channel == "coref_span" else None
    return row


def test_provenance_channel_labels_each_route():
    """The three routes a sentence can take into the target population: it named
    the company, or coref resolved it with a substitutable span, or coref
    resolved it without one. The span/no-span split is the point -- the two are
    audited at very different accuracy."""
    df = _frame(
        [
            _prov_row("A", 0, "surface"),
            _prov_row("A", 1, "coref_span"),
            _prov_row("A", 2, "coref_nospan"),
        ]
    )
    assert provenance_channel(df).tolist() == [
        "surface",
        "coref_span",
        "coref_nospan",
    ]


def test_provenance_channel_is_none_for_non_target_rows():
    df = _frame([_prov_row("A", 0, "surface", mentions_target=False)])
    assert provenance_channel(df).tolist() == [None]


def test_provenance_channel_defaults_to_surface_without_provenance_columns():
    """An older sentence table carries no resolution flags; every target row
    reading as `surface` is the honest answer for a table with no evidence
    that resolution ever happened."""
    df = _frame([_fus_row("A", 0, True, pos=0.5, neg=0.2, absa_pos=0.6, absa_neg=0.1)])
    assert provenance_channel(df).tolist() == ["surface"]


def test_channels_partition_the_target_population():
    """No sentence counted twice, none dropped -- the property that makes the
    counts safe to sum and the shares safe to read."""
    df = _frame(
        [
            _prov_row("A", 0, "surface"),
            _prov_row("A", 1, "coref_span"),
            _prov_row("A", 2, "coref_span"),
            _prov_row("A", 3, "coref_nospan"),
            _prov_row("A", 4, "surface", is_boilerplate=True),  # excluded
            _prov_row("A", 5, "surface", mentions_target=False),  # excluded
        ]
    )
    row = aggregate_provenance_features(df).set_index("article_id").loc["A"]
    counts = [row[f"prov_{c}_n"] for c in PROVENANCE_CHANNELS]
    assert counts == [1, 2, 1]
    assert sum(counts) == 4
    assert sum(row[f"prov_{c}_share"] for c in PROVENANCE_CHANNELS) == pytest.approx(1.0)


def test_provenance_does_not_change_the_blended_fusion_features():
    """The whole point of `purely additive`: aggregate_fusion_features() must
    return exactly what it returned before provenance existed."""
    df = _frame(
        [
            _prov_row("A", 0, "surface", pos=0.8, neg=0.1, absa_pos=0.9, absa_neg=0.05),
            _prov_row("A", 1, "coref_nospan", pos=0.2, neg=0.7, absa_pos=0.1, absa_neg=0.8),
            _prov_row("A", 2, "coref_span", pos=0.5, neg=0.4, absa_pos=0.5, absa_neg=0.4),
        ]
    )
    plain = df.drop(
        columns=["resolved_by_coref", "resolved_by_coref", "mention_char_start", "mention_char_end"]
    )
    with_provenance = aggregate_fusion_features(df).set_index("article_id")
    without = aggregate_fusion_features(plain).set_index("article_id")
    pd.testing.assert_frame_equal(with_provenance, without)


def test_channel_means_reproduce_the_blend_when_one_channel_holds_everything():
    """An article whose target rows are all one channel: that channel's mean
    must equal fus_<variant>_mean exactly."""
    df = _frame(
        [
            _prov_row("A", 0, "coref_nospan", pos=0.8, neg=0.1, absa_pos=0.9, absa_neg=0.05),
            _prov_row("A", 1, "coref_nospan", pos=0.2, neg=0.7, absa_pos=0.1, absa_neg=0.8),
        ]
    )
    blended = aggregate_fusion_features(df).set_index("article_id").loc["A"]
    split = aggregate_provenance_features(df).set_index("article_id").loc["A"]
    for variant in AGGREGATED_VARIANTS:
        assert split[f"prov_coref_nospan_{variant}_mean"] == pytest.approx(
            blended[f"fus_{variant}_mean"]
        )


def test_channel_means_are_computed_over_only_that_channels_sentences():
    df = _frame(
        [
            _prov_row("A", 0, "surface", pos=0.9, neg=0.0, absa_pos=1.0, absa_neg=0.0),
            _prov_row("A", 1, "coref_nospan", pos=0.0, neg=0.9, absa_pos=0.0, absa_neg=1.0),
        ]
    )
    row = aggregate_provenance_features(df).set_index("article_id").loc["A"]
    # conf_graft = absa * |fin|: surface = +1.0 * 0.9, coref_nospan = -1.0 * 0.9
    assert row["prov_surface_conf_graft_floor_mean"] == pytest.approx(0.9)
    assert row["prov_coref_nospan_conf_graft_floor_mean"] == pytest.approx(-0.9)


def test_empty_channel_is_nan_for_sentiment_and_zero_for_count():
    """no-signal-vs-measured-neutral: a channel with no sentences measured
    nothing (NaN), but its count of zero is a real measurement."""
    df = _frame([_prov_row("A", 0, "surface")])
    row = aggregate_provenance_features(df).set_index("article_id").loc["A"]
    assert row["prov_coref_nospan_n"] == 0
    for variant in AGGREGATED_VARIANTS:
        assert pd.isna(row[f"prov_coref_nospan_{variant}_mean"])


def test_article_with_no_target_sentences_has_nan_shares_and_zero_counts():
    df = _frame([_prov_row("A", 0, "surface", mentions_target=False)])
    row = aggregate_provenance_features(df).set_index("article_id").loc["A"]
    for channel in PROVENANCE_CHANNELS:
        assert row[f"prov_{channel}_n"] == 0
        assert pd.isna(row[f"prov_{channel}_share"])


def test_boilerplate_rows_are_excluded_from_every_channel():
    df = _frame(
        [
            _prov_row("A", 0, "coref_span"),
            _prov_row("A", 1, "coref_span", is_boilerplate=True),
        ]
    )
    row = aggregate_provenance_features(df).set_index("article_id").loc["A"]
    assert row["prov_coref_span_n"] == 1


def test_provenance_empty_input_returns_empty_frame_with_correct_columns():
    df = pd.DataFrame(
        columns=[
            "article_id",
            "sent_idx",
            "mentions_target",
            "is_boilerplate",
            "pos",
            "neg",
            "absa_pos",
            "absa_neg",
            "absa_neu",
        ]
    )
    result = aggregate_provenance_features(df)
    assert len(result) == 0
    expected = ["article_id"]
    for channel in PROVENANCE_CHANNELS:
        expected.append(f"prov_{channel}_n")
        expected.append(f"prov_{channel}_share")
        expected.extend(f"prov_{channel}_{variant}_mean" for variant in AGGREGATED_VARIANTS)
    assert list(result.columns) == expected
