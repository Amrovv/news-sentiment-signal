"""Tests for the referent-verification judge.

None of these load the 4.7GB GGUF. The prompt is the part a human can be wrong
about, so it is the part pinned here; inference itself is llama.cpp's problem.
"""

import pandas as pd
import pytest

from news_sentiment.text.coref_eval import JUDGE_ANSWERS, accept_only
from news_sentiment.text.coref_judge import (
    ANSWERS,
    PROMPT_VERSION,
    build_prompt,
    company_name,
    judge_corpus,
    judge_population,
    make_judge,
)


def _ctx(sentence="The company grew.", span=(0, 11), headline="A headline"):
    from news_sentiment.text.coref_eval import JudgeContext

    return JudgeContext(
        article_id=1,
        sent_idx=3,
        target="TSLA",
        headline=headline,
        preceding=("Before one.", "Before two."),
        sentence=sentence,
        following=("After one.",),
        mention_char_start=None if span is None else span[0],
        mention_char_end=None if span is None else span[1],
    )


# ---------------------------------------------------------------------------
# the answer vocabulary must match the harness's contract
# ---------------------------------------------------------------------------


def test_judge_answers_match_the_harness_contract():
    """If these drift apart, a valid answer becomes an unparseable one and the
    judge silently discards rows it meant to accept."""
    assert set(ANSWERS) == set(JUDGE_ANSWERS)


def test_only_yes_is_accepted_by_the_harness():
    assert accept_only("yes")
    assert not accept_only("no")
    assert not accept_only("unsure")


# ---------------------------------------------------------------------------
# ticker-agnosticism
# ---------------------------------------------------------------------------


def test_company_name_comes_from_config_not_a_literal():
    assert company_name("TSLA") == "Tesla"


def test_company_name_falls_back_to_the_ticker_for_an_unknown_target():
    assert company_name("ZZZZ") == "ZZZZ"


def test_prompt_names_the_target_company_and_no_other():
    """No roster of confusable companies anywhere in the prompt -- the whole
    point is that the model supplies world knowledge instead."""
    prompt = build_prompt(_ctx(), target="TSLA")
    assert "Tesla" in prompt
    for confusor in ("Rivian", "SpaceX", "BYD", "Lucid", "NIO"):
        assert confusor not in prompt


def test_prompt_uses_the_target_passed_in():
    prompt = build_prompt(_ctx(), target="NVDA")
    assert "Nvidia" in prompt or "NVDA" in prompt
    assert "Tesla" not in prompt


# ---------------------------------------------------------------------------
# the prompt itself
# ---------------------------------------------------------------------------


def test_span_prompt_asks_about_the_marked_phrase():
    prompt = build_prompt(_ctx())
    assert "marked phrase" in prompt
    assert "**The company**" in prompt


def test_no_span_prompt_asks_about_the_sentence_instead():
    """A no-span row has no marked phrase; asking about one would be incoherent
    and would score the judge on a question it was never asked."""
    prompt = build_prompt(_ctx(span=None))
    assert "marked phrase" not in prompt
    assert "marked sentence" in prompt


def test_prompt_offers_exactly_the_three_answers():
    prompt = build_prompt(_ctx())
    assert "yes, no, or unsure" in prompt


def test_prompt_includes_the_full_context_window():
    prompt = build_prompt(_ctx())
    for fragment in ("A headline", "Before one.", "Before two.", "After one."):
        assert fragment in prompt


def test_prompt_version_is_set():
    """It is part of the verdict cache key; an empty one would collide every
    prompt revision into the same cached answers."""
    assert PROMPT_VERSION and isinstance(PROMPT_VERSION, str)


# ---------------------------------------------------------------------------
# fail-closed behaviour with no model present
# ---------------------------------------------------------------------------


def test_judge_returns_unsure_when_the_model_file_is_absent(tmp_path, monkeypatch):
    """A missing model must discard every row, not raise. Reset the module's
    memoised handle so this test does not depend on execution order."""
    import news_sentiment.text.coref_judge as cj

    monkeypatch.setattr(cj, "_MODEL", None)
    monkeypatch.setattr(cj, "_LOAD_FAILED", False)

    judge = make_judge(model_path=tmp_path / "nonexistent.gguf")
    assert judge(_ctx()) == "unsure"
    assert not accept_only(judge(_ctx()))


def test_missing_model_is_not_reported_per_row(tmp_path, monkeypatch, caplog):
    """_LOAD_FAILED latches, so a 3-hour run does not emit one warning per row."""
    import news_sentiment.text.coref_judge as cj

    monkeypatch.setattr(cj, "_MODEL", None)
    monkeypatch.setattr(cj, "_LOAD_FAILED", False)

    judge = make_judge(model_path=tmp_path / "nonexistent.gguf")
    for _ in range(5):
        judge(_ctx())
    assert cj._LOAD_FAILED is True


@pytest.mark.parametrize("span", [(0, 11), None])
def test_prompt_builds_for_both_populations(span):
    prompt = build_prompt(_ctx(span=span))
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "Tesla" in prompt


# ---------------------------------------------------------------------------
# industry neutrality -- ticker-agnostic means the prompt must not encode the
# target's SECTOR any more than it encodes the target's competitors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word",
    [
        "vehicle",   # v1 said "products or vehicles" -- an automaker assumption
        "car",
        "automaker",
        "electric",
        "EV",
        "chip",
        "beverage",
        "retail",
        "drug",
    ],
)
def test_prompt_contains_no_sector_vocabulary(word):
    """The prompt is shown verbatim for TSLA, NVDA or KO alike. Any sector noun
    in it is a hint that exists only because of which ticker we happen to be
    pointed at, which is the ticker-agnostic constraint being violated quietly."""
    # Whole words only: "car" lives inside "careful" and "EV" inside "evidence",
    # and a naive substring check would fail on both without anything having
    # leaked. Only the standalone noun is the violation.
    import re

    for span in [(0, 11), None]:
        prompt = build_prompt(_ctx(span=span), target="TSLA")
        pattern = rf"\b{re.escape(word)}s?\b"
        assert not re.search(pattern, prompt, flags=re.IGNORECASE), (
            f"sector word {word!r} leaked into the prompt"
        )


def test_target_own_products_are_accepted_not_rejected():
    """Convention: sentiment about a company's own product is sentiment about
    the company. The prompt must say so explicitly in both populations."""
    for span in [(0, 11), None]:
        prompt = build_prompt(_ctx(span=span))
        yes_clause = prompt.split('Answer "no"')[0]
        assert "own products" in yes_clause


def test_other_companies_products_are_still_rejected():
    for span in [(0, 11), None]:
        prompt = build_prompt(_ctx(span=span))
        no_clause = prompt.split('Answer "no"')[1]
        assert "different company's products" in no_clause


# ---------------------------------------------------------------------------
# the judge callable itself must fail closed, not only evaluate_judge()
# ---------------------------------------------------------------------------


def test_judge_returns_unsure_when_inference_raises(monkeypatch):
    """REGRESSION: a prompt exceeding n_ctx raised ValueError out of llama.cpp
    and killed a 3.3-hour corpus pass at 67% complete. The module contract says
    outright exceptions are discards, but only evaluate_judge() delivered that,
    so any caller driving the judge directly -- which is what a corpus run does
    -- inherited a crash instead of a discard."""
    import news_sentiment.text.coref_judge as cj

    class Exploding:
        def __call__(self, *a, **k):
            raise ValueError("Requested tokens (2316) exceed context window of 2048")

    monkeypatch.setattr(cj, "_MODEL", Exploding())
    monkeypatch.setattr(cj, "_LOAD_FAILED", False)
    monkeypatch.setattr(cj, "_grammar", lambda: None)

    judge = cj.make_judge()
    assert judge(_ctx()) == "unsure"
    assert not accept_only(judge(_ctx()))


def test_default_context_window_covers_the_measured_corpus_maximum():
    """Corpus max prompt is 2,446 tokens; the 270-row eval set topped out at 513
    and gave no warning. The default must clear the corpus maximum, not the
    sample's."""
    from news_sentiment.text.coref_judge import DEFAULT_N_CTX

    assert DEFAULT_N_CTX >= 2446


# ---------------------------------------------------------------------------
# judge_corpus: the batch entry point
# ---------------------------------------------------------------------------


def _corpus(n_boilerplate=0):
    """A tiny sentence table plus its article table.

    Row 0 names the company (surface, never judged), row 1 is coref-resolved
    (judged), row 2 is coref-resolved boilerplate (outside the population).
    """
    sentences = pd.DataFrame({
        "article_id": [1, 1, 1],
        "sent_idx": [0, 1, 2],
        "text": ["Tesla raised prices.", "The company also grew.", "Image source: Getty."],
        "mentions_target": [True, True, True],
        "is_boilerplate": [False, False, True],
        "resolved_by_coref": [False, True, True],
        "mention_char_start": pd.array([None, 0, None], dtype="Int64"),
        "mention_char_end": pd.array([None, 11, None], dtype="Int64"),
    })
    articles = pd.DataFrame({"article_id": [1], "headline": ["A headline"]})
    return sentences, articles


def test_judge_population_is_coref_target_non_boilerplate():
    sentences, _ = _corpus()
    mask = judge_population(sentences)
    assert list(mask) == [False, True, False]


def test_judge_population_treats_missing_columns_as_nothing_to_judge():
    """An older sentence table must degrade to an empty population, not raise."""
    bare = pd.DataFrame({"article_id": [1], "sent_idx": [0], "text": ["x"]})
    assert not judge_population(bare).any()


def test_judge_corpus_judges_only_the_population(tmp_path):
    sentences, articles = _corpus()
    seen = []

    def spy(ctx):
        seen.append((ctx.article_id, ctx.sent_idx))
        return "yes"

    out = judge_corpus(
        sentences, articles, judge=spy, cache_path=tmp_path / "judge.parquet"
    )
    assert seen == [(1, 1)]
    # NA outside the population: nothing was asked, so there is no answer.
    assert out["judge_answer"].notna().tolist() == [False, True, False]
    assert out.loc[1, "judge_answer"] == "yes"


def test_judge_corpus_gate_only_rejects_rows_it_asked_about(tmp_path):
    """A surface match and a boilerplate row pass because the judge has no
    opinion on them; only a judged row can fail."""
    sentences, articles = _corpus()
    out = judge_corpus(
        sentences, articles, judge=lambda ctx: "no", cache_path=tmp_path / "judge.parquet"
    )
    assert out["judge_accepted"].tolist() == [True, False, True]


@pytest.mark.parametrize("answer", ["no", "unsure", "YES!", "", "maybe"])
def test_judge_corpus_fails_closed_on_anything_but_yes(answer, tmp_path):
    sentences, articles = _corpus()
    out = judge_corpus(
        sentences, articles, judge=lambda ctx: answer, cache_path=tmp_path / "judge.parquet"
    )
    assert not out.loc[1, "judge_accepted"]


def test_judge_corpus_resumes_from_cache_without_calling_the_judge(tmp_path):
    """The whole point of the cache: a re-run must not re-judge an answered row,
    which is what makes an interrupted multi-hour pass survivable."""
    sentences, articles = _corpus()
    cache_path = tmp_path / "judge.parquet"
    judge_corpus(sentences, articles, judge=lambda ctx: "yes", cache_path=cache_path)

    def exploding(ctx):
        raise AssertionError("re-judged a row that was already cached")

    out = judge_corpus(sentences, articles, judge=exploding, cache_path=cache_path)
    assert out.loc[1, "judge_answer"] == "yes"
    assert out.loc[1, "judge_accepted"]


def test_judge_corpus_does_not_reuse_verdicts_across_prompt_versions(tmp_path):
    """prompt_version is in the cache key so an edited prompt cannot be scored
    with the old prompt's answers."""
    sentences, articles = _corpus()
    cache_path = tmp_path / "judge.parquet"
    judge_corpus(
        sentences, articles, judge=lambda ctx: "yes",
        prompt_version="v1", cache_path=cache_path,
    )
    out = judge_corpus(
        sentences, articles, judge=lambda ctx: "no",
        prompt_version="v2", cache_path=cache_path,
    )
    assert out.loc[1, "judge_answer"] == "no"


def test_judge_corpus_leaves_the_input_frame_untouched(tmp_path):
    sentences, articles = _corpus()
    before = sentences.copy()
    judge_corpus(
        sentences, articles, judge=lambda ctx: "yes", cache_path=tmp_path / "judge.parquet"
    )
    pd.testing.assert_frame_equal(sentences, before)
