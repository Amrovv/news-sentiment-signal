import numpy as np
import pandas as pd
import pytest

from stock_predictor.text.coref_eval import (
    CACHE_COLUMNS,
    JudgeContext,
    accept_only,
    build_context,
    build_contexts,
    evaluate_judge,
    load_eval_set,
    load_judge_cache,
    save_judge_cache,
    verify_contexts_match,
    wilson_ci,
)


def _sentences(article_id=1, n=10):
    return pd.DataFrame(
        {
            "article_id": [article_id] * n,
            "sent_idx": list(range(n)),
            "text": [f"Sentence {i}." for i in range(n)],
        }
    )


def _articles(article_id=1, headline="A headline"):
    return pd.DataFrame({"article_id": [article_id], "headline": [headline]})


def _eval_frame(rows):
    """Minimal eval frame with the columns evaluate_judge() reads."""
    return pd.DataFrame(rows)


def _row(row_id, sent_idx, verdict, has_span=True, borderline=False, article_id=1, span=None):
    return {
        "row_id": row_id,
        "article_id": article_id,
        "sent_idx": sent_idx,
        "text": f"Sentence {sent_idx}.",
        "has_span": has_span,
        "verdict": verdict,
        "borderline": borderline,
        "mention_char_start": None if span is None else span[0],
        "mention_char_end": None if span is None else span[1],
    }


# ---------------------------------------------------------------------------
# wilson_ci -- must agree with the definition notebook 2.7 uses
# ---------------------------------------------------------------------------


def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = wilson_ci(89, 100)
    assert lo < 0.89 < hi


def test_wilson_ci_stays_inside_zero_one_at_the_extremes():
    # Tolerance, not clamping: at k=0 (and k=n) the closed form lands on 0 (or 1)
    # up to float rounding, and wilson_ci is kept byte-identical to notebook 2.7's
    # definition so numbers computed here and there are directly comparable.
    # Clamping would break that identity to hide an error of 3e-17.
    eps = 1e-12
    lo, hi = wilson_ci(0, 10)
    assert lo >= -eps and hi <= 1.0 + eps
    lo, hi = wilson_ci(10, 10)
    assert lo >= -eps and hi <= 1.0 + eps


def test_wilson_ci_is_nan_for_empty_sample():
    lo, hi = wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_wilson_ci_narrows_as_n_grows():
    narrow = wilson_ci(85, 170)
    wide = wilson_ci(25, 50)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# ---------------------------------------------------------------------------
# accept_only -- the fail-closed rule, in one place
# ---------------------------------------------------------------------------


def test_accept_only_accepts_yes_case_insensitively():
    assert accept_only("yes")
    assert accept_only("YES")
    assert accept_only("  yes  ")


@pytest.mark.parametrize("answer", ["no", "unsure", "", "maybe", "yes, definitely", None, 1])
def test_accept_only_discards_everything_else(answer):
    assert not accept_only(answer)


# ---------------------------------------------------------------------------
# build_context -- the shared definition of "what gets read"
# ---------------------------------------------------------------------------


def test_build_context_uses_four_preceding_and_one_following():
    ctx = build_context(1, 5, _sentences(), {1: "A headline"})
    assert ctx.preceding == ("Sentence 1.", "Sentence 2.", "Sentence 3.", "Sentence 4.")
    assert ctx.sentence == "Sentence 5."
    assert ctx.following == ("Sentence 6.",)
    assert ctx.headline == "A headline"


def test_build_context_truncates_at_article_boundaries():
    ctx = build_context(1, 0, _sentences(n=2), {1: "H"})
    assert ctx.preceding == ()
    assert ctx.following == ("Sentence 1.",)


def test_build_context_raises_on_missing_sentence():
    with pytest.raises(KeyError):
        build_context(1, 99, _sentences(), {1: "H"})


def test_build_context_treats_nan_span_as_no_span():
    ctx = build_context(1, 3, _sentences(), {1: "H"}, mention_char_start=np.nan)
    assert not ctx.has_span
    assert ctx.mention is None


def test_build_contexts_preserves_eval_frame_order():
    eval_df = _eval_frame([_row("R-1", 7, "target"), _row("R-2", 2, "other")])
    contexts = build_contexts(eval_df, _sentences(), _articles())
    assert [c.sent_idx for c in contexts] == [7, 2]


# ---------------------------------------------------------------------------
# JudgeContext rendering -- the judge must be told WHICH mention is in question
# ---------------------------------------------------------------------------


def test_marked_sentence_delimits_the_mention():
    ctx = JudgeContext(1, 0, "TSLA", "H", (), "The company grew.", (), 0, 11)
    assert ctx.marked_sentence() == "**The company** grew."
    assert ctx.mention == "The company"


def test_marked_sentence_is_unchanged_without_a_span():
    ctx = JudgeContext(1, 0, "TSLA", "H", (), "It grew.", ())
    assert ctx.marked_sentence() == "It grew."


def test_render_includes_headline_context_and_the_marked_sentence():
    ctx = JudgeContext(1, 0, "TSLA", "Headline", ("Before.",), "It grew.", ("After.",))
    rendered = ctx.render()
    assert "HEADLINE: Headline" in rendered
    assert "Before." in rendered and "After." in rendered
    assert "> It grew." in rendered


# ---------------------------------------------------------------------------
# verify_contexts_match -- corpus/label drift must be loud, not silent
# ---------------------------------------------------------------------------


def test_verify_contexts_match_passes_when_text_agrees():
    # Asserts by not raising: this is the happy path of the drift check that
    # evaluate_judge() runs before any judging.
    eval_df = _eval_frame([_row("R-1", 3, "target")])
    verify_contexts_match(eval_df, build_contexts(eval_df, _sentences(), _articles()))


def test_verify_contexts_match_raises_on_drift():
    eval_df = _eval_frame([_row("R-1", 3, "target")])
    eval_df.loc[0, "text"] = "A different sentence than the corpus has."
    contexts = build_contexts(eval_df, _sentences(), _articles())
    with pytest.raises(ValueError, match="do not match the labelled text"):
        verify_contexts_match(eval_df, contexts)


# ---------------------------------------------------------------------------
# evaluate_judge -- stub judges with hand-computable metrics
# ---------------------------------------------------------------------------


def _mixed_frame():
    """6 rows: span 2 target / 1 error, no-span 1 target / 2 errors."""
    return _eval_frame(
        [
            _row("S-1", 0, "target", has_span=True, span=(0, 8)),
            _row("S-2", 1, "target", has_span=True, span=(0, 8)),
            _row("S-3", 2, "other", has_span=True, span=(0, 8)),
            _row("N-1", 3, "target", has_span=False),
            _row("N-2", 4, "other", has_span=False),
            _row("N-3", 5, "other", has_span=False),
        ]
    )


def _evaluate(judge, eval_df=None, **kwargs):
    eval_df = _mixed_frame() if eval_df is None else eval_df
    return evaluate_judge(
        judge,
        eval_df,
        sentences=_sentences(),
        articles=_articles(),
        use_cache=False,
        **kwargs,
    )


def _slice(results, judge, population, borderline="included"):
    hit = results[
        (results["judge"] == judge)
        & (results["population"] == population)
        & (results["borderline"] == borderline)
    ]
    assert len(hit) == 1, f"expected one row, got {len(hit)}"
    return hit.iloc[0]


def test_accept_all_judge_reproduces_the_baseline():
    results = _evaluate(lambda ctx: "yes", model_id="always-yes")
    for population in ("span", "no-span", "all"):
        judged = _slice(results, "always-yes", population)
        baseline = _slice(results, "baseline (accept all)", population)
        assert judged["accept_precision"] == pytest.approx(baseline["accept_precision"])
        assert judged["correct_lost"] == 0


def test_reject_all_judge_catches_every_error_and_loses_every_correct_row():
    results = _evaluate(lambda ctx: "no", model_id="always-no")
    row = _slice(results, "always-no", "all")
    assert row["n_accepted"] == 0
    assert row["error_recall"] == pytest.approx(1.0)
    assert row["correct_lost"] == 3
    assert np.isnan(row["accept_precision"])


def test_oracle_judge_reaches_perfect_accept_precision_with_no_loss():
    labels = {r["sent_idx"]: r["verdict"] for r in _mixed_frame().to_dict("records")}
    oracle = lambda ctx: "yes" if labels[ctx.sent_idx] == "target" else "no"  # noqa: E731

    results = _evaluate(oracle, model_id="oracle")
    row = _slice(results, "oracle", "all")
    assert row["accept_precision"] == pytest.approx(1.0)
    assert row["error_recall"] == pytest.approx(1.0)
    assert row["correct_lost"] == 0
    assert row["n_accepted"] == 3


def test_metrics_are_reported_separately_per_population():
    results = _evaluate(lambda ctx: "yes", model_id="always-yes")
    assert _slice(results, "baseline (accept all)", "span")["baseline_precision"] == pytest.approx(2 / 3)
    assert _slice(results, "baseline (accept all)", "no-span")["baseline_precision"] == pytest.approx(1 / 3)
    assert _slice(results, "baseline (accept all)", "all")["baseline_precision"] == pytest.approx(0.5)


def test_unsure_is_a_discard_not_a_third_outcome():
    results = _evaluate(lambda ctx: "unsure", model_id="always-unsure")
    assert _slice(results, "always-unsure", "all")["n_accepted"] == 0


def test_unrecognised_answer_is_treated_as_unsure():
    results = _evaluate(lambda ctx: "probably not", model_id="chatty")
    assert _slice(results, "chatty", "all")["n_accepted"] == 0


def test_judge_exception_fails_closed_rather_than_propagating():
    def explodes(ctx):
        raise RuntimeError("model died")

    results = _evaluate(explodes, model_id="broken")
    assert _slice(results, "broken", "all")["n_accepted"] == 0


def test_borderline_rows_are_reported_both_ways():
    eval_df = _mixed_frame()
    eval_df.loc[eval_df["row_id"] == "S-1", "borderline"] = True

    results = _evaluate(lambda ctx: "yes", eval_df=eval_df, model_id="always-yes")
    assert _slice(results, "baseline (accept all)", "all", "included")["n"] == 6
    assert _slice(results, "baseline (accept all)", "all", "excluded")["n"] == 5


def test_evaluate_judge_rejects_mismatched_context_count():
    eval_df = _mixed_frame()
    contexts = build_contexts(eval_df, _sentences(), _articles())[:-1]
    with pytest.raises(ValueError, match="contexts for"):
        evaluate_judge(lambda ctx: "yes", eval_df, contexts=contexts, use_cache=False)


def test_evaluate_judge_requires_contexts_or_the_frames_to_build_them():
    with pytest.raises(ValueError, match="Pass either"):
        evaluate_judge(lambda ctx: "yes", _mixed_frame(), use_cache=False)


# ---------------------------------------------------------------------------
# the verdict cache -- merge, never replace
# ---------------------------------------------------------------------------


def test_judge_cache_round_trips(tmp_path):
    path = tmp_path / "judge_cache.parquet"
    rows = pd.DataFrame(
        [{"article_id": 1, "sent_idx": 0, "target": "TSLA", "model_id": "m", "prompt_version": "v1", "answer": "yes"}],
        columns=CACHE_COLUMNS,
    )
    save_judge_cache(rows, path=path)
    assert load_judge_cache(path)["answer"].tolist() == ["yes"]


def test_missing_judge_cache_is_empty_not_an_error(tmp_path):
    assert load_judge_cache(tmp_path / "absent.parquet").empty


def test_unreadable_judge_cache_degrades_to_empty(tmp_path):
    path = tmp_path / "corrupt.parquet"
    path.write_text("not a parquet file")
    assert load_judge_cache(path).empty


def test_judge_cache_dedupes_on_the_full_key_keeping_last(tmp_path):
    path = tmp_path / "judge_cache.parquet"
    key = {"article_id": 1, "sent_idx": 0, "target": "TSLA", "model_id": "m"}
    rows = pd.DataFrame(
        [
            {**key, "prompt_version": "v1", "answer": "yes"},
            {**key, "prompt_version": "v1", "answer": "no"},
            {**key, "prompt_version": "v2", "answer": "yes"},
        ],
        columns=CACHE_COLUMNS,
    )
    save_judge_cache(rows, path=path)
    loaded = load_judge_cache(path)
    # v1 collapses to its last value; v2 is a different key and survives.
    assert len(loaded) == 2
    assert loaded[loaded["prompt_version"] == "v1"]["answer"].tolist() == ["no"]


def test_prompt_version_change_does_not_reuse_old_verdicts(tmp_path):
    """The whole reason prompt_version is in the cache key."""
    path = tmp_path / "judge_cache.parquet"
    eval_df = _mixed_frame()
    calls = []

    def counting_judge(ctx):
        calls.append(ctx.sent_idx)
        return "yes"

    kwargs = dict(sentences=_sentences(), articles=_articles(), cache_path=path, model_id="m")
    evaluate_judge(counting_judge, eval_df, prompt_version="v1", **kwargs)
    first = len(calls)
    evaluate_judge(counting_judge, eval_df, prompt_version="v1", **kwargs)
    assert len(calls) == first, "second run under the same prompt_version should be all cache hits"
    evaluate_judge(counting_judge, eval_df, prompt_version="v2", **kwargs)
    assert len(calls) == first * 2, "a new prompt_version must re-judge every row"


# ---------------------------------------------------------------------------
# the real eval set
# ---------------------------------------------------------------------------


def test_real_eval_set_loads_with_the_expected_shape():
    df = load_eval_set()
    assert len(df) == 270
    assert int(df["has_span"].sum()) == 100
    # 63, after two convention changes on 2026-08-18 (see the test below).
    # It was 85 for most of this branch's life, then 84, then 63.
    assert int((df["verdict"] == "other").sum()) == 63


def test_real_eval_set_baseline_matches_the_published_figures():
    """90.0% span / 68.8% no-span -- notebook 2.7 §2. A drift here means the
    labels changed and every figure in 2.7 needs recomputing.

    TWO CONVENTION CHANGES ON 2026-08-18 moved these off the 89.0% / 56.5% (85
    errors) this branch reported for most of its life:

      1. A reference to one of the TARGET's own products counts as the company,
         since sentiment about a company's product is sentiment about the
         company for a price model. Moved 1 span row (SPAN-16, the Semi).
      2. A joint referent that includes the target ("both firms", "the two
         global EV leaders"), a fund or basket holding the target, and a generic
         or third-party statement made in a target article all read as direct
         references to the target. Moved 21 no-span rows.

    Inverse instruments are the deliberate exception and remain errors: TSLQ's
    sentiment is sign-flipped, so accepting it inverts the signal rather than
    diluting it."""
    df = load_eval_set()
    span = df[df["has_span"]]
    nospan = df[~df["has_span"]]
    assert (span["verdict"] == "target").mean() == pytest.approx(0.900)
    assert (nospan["verdict"] == "target").mean() == pytest.approx(0.688, abs=0.001)
