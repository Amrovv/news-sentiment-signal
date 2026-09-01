import pandas as pd
import pytest

from news_sentiment.text import absa
from news_sentiment.text.sentiment import ABSA_FEATURE_COLUMNS, aggregate_article_features

TICKER = "TSLA"


def _sentence_row(
    text,
    *,
    article_id=1,
    sent_idx=0,
    mentions_target=False,
    mentions_ceo=False,
    resolved_by_coref=False,
    is_boilerplate=False,
    mention_char_start=None,
    mention_char_end=None,
):
    """One row of the entity_filter sentence schema, spelled out by hand.

    Hand-built rather than produced by process_articles() so each pair-building
    rule can be exercised in isolation, without a spaCy parse deciding which
    branch is under test.
    """
    return {
        "article_id": article_id,
        "sent_idx": sent_idx,
        "text": text,
        "mentions_target": mentions_target,
        "mentions_ceo": mentions_ceo,
        "resolved_by_coref": resolved_by_coref,
        "is_boilerplate": is_boilerplate,
        "char_len": len(text),
        "mention_char_start": mention_char_start,
        "mention_char_end": mention_char_end,
    }


def _frame(rows):
    df = pd.DataFrame(rows)
    return df.astype({"mention_char_start": "Int64", "mention_char_end": "Int64"})


# ---------------------------------------------------------------------------
# pair_hash
# ---------------------------------------------------------------------------


def test_pair_hash_keys_on_text_and_aspect_together():
    text = "Build-A-Bear outperformed Tesla this quarter."
    assert absa.pair_hash(text, "Tesla") != absa.pair_hash(text, "Build-A-Bear")
    assert absa.pair_hash(text, "Tesla") == absa.pair_hash(text, "Tesla")
    # Whitespace is stripped on both halves, like sentiment.hash_text.
    assert absa.pair_hash(f"  {text} ", " Tesla ") == absa.pair_hash(text, "Tesla")


# ---------------------------------------------------------------------------
# build_pairs
# ---------------------------------------------------------------------------


def test_build_pairs_explicit_target_scores_text_unchanged():
    text = "Tesla delivered record numbers this quarter."
    out = absa.build_pairs(_frame([_sentence_row(text, mentions_target=True)]), TICKER)

    assert out.loc[0, "absa_text"] == text
    assert out.loc[0, "absa_aspect"] == "Tesla"


def test_build_pairs_substitutes_resolved_mention_and_leaves_text_intact():
    # An ABSA model attends to an aspect TERM present in the text. An referring
    # sentence names no company, so the resolved name is injected over the
    # mention's own characters -- that is what mention_char_start/end are for.
    text = "The company also raised prices across its lineup."
    start, end = 0, len("The company")
    rows = [
        _sentence_row(
            text,
            mentions_target=True,
            resolved_by_coref=True,
            mention_char_start=start,
            mention_char_end=end,
        )
    ]
    out = absa.build_pairs(_frame(rows), TICKER)

    assert out.loc[0, "absa_text"] == "Tesla also raised prices across its lineup."
    assert "Tesla" in out.loc[0, "absa_text"]
    assert out.loc[0, "absa_aspect"] == "Tesla"
    # The ORIGINAL text column must never be rewritten: every FinBERT feature
    # and every downstream inspection still reads the sentence as published.
    assert out.loc[0, "text"] == text


def test_build_pairs_substitutes_coref_resolved_span():
    text = "It expects deliveries to keep climbing next year."
    out = absa.build_pairs(
        _frame(
            [
                _sentence_row(
                    text,
                    mentions_target=True,
                    resolved_by_coref=True,
                    mention_char_start=0,
                    mention_char_end=2,
                )
            ]
        ),
        TICKER,
    )
    assert out.loc[0, "absa_text"] == "Tesla expects deliveries to keep climbing next year."
    assert out.loc[0, "absa_aspect"] == "Tesla"


@pytest.mark.parametrize(
    "start,end",
    [
        (None, None),  # resolved flag set but no span recorded
        (5, 3),  # reversed
        (0, 10_000),  # past the end of the sentence
        (-1, 4),  # negative
    ],
)
def test_build_pairs_invalid_span_falls_back_to_unchanged_text(start, end):
    # A bad offset is a one-sentence data-quality event, never an exception and
    # never a lost row.
    text = "The company also raised prices."
    out = absa.build_pairs(
        _frame(
            [
                _sentence_row(
                    text,
                    mentions_target=True,
                    resolved_by_coref=True,
                    mention_char_start=start,
                    mention_char_end=end,
                )
            ]
        ),
        TICKER,
    )
    assert out.loc[0, "absa_text"] == text
    assert out.loc[0, "absa_aspect"] == "Tesla"


def test_build_pairs_ceo_only_uses_person_alias_and_never_substitutes():
    # Musk is a legitimate aspect in his own right; the person tier has never
    # been allowed to speak for the company, and substitution here would make it
    # do exactly that.
    text = "Musk said the case would proceed."
    out = absa.build_pairs(_frame([_sentence_row(text, mentions_ceo=True)]), TICKER)
    assert out.loc[0, "absa_text"] == text
    assert out.loc[0, "absa_aspect"] == "Musk"


def test_build_pairs_skips_rows_needs_score_rejects():
    rows = [
        # In no bucket at all.
        _sentence_row("The weather in Austin was unremarkable.", sent_idx=0),
        # Boilerplate, even though it names the target.
        _sentence_row(
            "Tesla investors should read the disclosure.",
            sent_idx=1,
            mentions_target=True,
            is_boilerplate=True,
        ),
    ]
    out = absa.build_pairs(_frame(rows), TICKER)
    assert list(out["absa_text"]) == ["", ""]
    assert list(out["absa_aspect"]) == ["", ""]


def test_build_pairs_comparative_row_takes_the_target_aspect():
    # One pair per row; reading comparatives correctly FOR THE TARGET is the
    # whole motivation for this module, so the target aspect wins.
    text = "Build-A-Bear outperformed Tesla this quarter."
    out = absa.build_pairs(
        _frame([_sentence_row(text, mentions_target=True)]),
        TICKER,
    )
    assert out.loc[0, "absa_aspect"] == "Tesla"


def test_build_pairs_empty_frame_gets_the_columns():
    out = absa.build_pairs(_frame([_sentence_row("x" * 30, mentions_target=True)]).iloc[0:0], TICKER)
    assert "absa_text" in out.columns
    assert "absa_aspect" in out.columns
    assert len(out) == 0


# ---------------------------------------------------------------------------
# score_pairs -- cache-hit-only calls must not load a model
# (mirrors test_score_sentences_all_cache_hits_requires_no_model)
# ---------------------------------------------------------------------------


def _cache_frame(pairs, pos=0.5, neg=0.3, neu=0.2):
    return pd.DataFrame(
        {
            "pair_hash": [absa.pair_hash(t, a) for t, a in pairs],
            "pos": [pos] * len(pairs),
            "neg": [neg] * len(pairs),
            "neu": [neu] * len(pairs),
        }
    )


def test_score_pairs_all_cache_hits_requires_no_model():
    pairs = [("Tesla beat expectations.", "Tesla"), ("Ford raised guidance.", "Ford")]
    cache_df = _cache_frame(pairs, pos=0.8, neg=0.1, neu=0.1)

    # model=None, tokenizer=None -- must not attempt to load a real model.
    result = absa.score_pairs(pairs, model=None, tokenizer=None, cache_df=cache_df)

    assert len(result) == 2
    assert set(result["pair_hash"]) == set(cache_df["pair_hash"])
    assert result["pos"].tolist() == [pytest.approx(0.8)] * 2


def test_score_pairs_dedupes_repeated_pairs():
    pairs = [("Tesla beat expectations.", "Tesla")] * 3
    cache_df = _cache_frame(pairs[:1])
    result = absa.score_pairs(pairs, cache_df=cache_df)
    assert len(result) == 1


def test_score_sentence_table_all_cache_hits_requires_no_model(tmp_path):
    text = "Tesla delivered record numbers this quarter."
    cache_path = tmp_path / "absa_cache.parquet"
    absa.save_cache(_cache_frame([(text, "Tesla")], pos=0.7, neg=0.2, neu=0.1), path=cache_path)

    out = absa.score_sentence_table(
        _frame([_sentence_row(text, mentions_target=True)]), TICKER, cache_path=cache_path
    )

    assert out.loc[0, "absa_pos"] == pytest.approx(0.7)
    # Rows with no pair stay NaN, never 0.
    out2 = absa.score_sentence_table(
        _frame(
            [
                _sentence_row(text, sent_idx=0, mentions_target=True),
                _sentence_row("The weather in Austin was unremarkable.", sent_idx=1),
            ]
        ),
        TICKER,
        cache_path=cache_path,
    )
    assert pd.isna(out2.loc[1, "absa_pos"])


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path):
    path = tmp_path / "absa_cache.parquet"
    assert len(absa.load_cache(path)) == 0

    frame = _cache_frame([("Tesla beat expectations.", "Tesla")], pos=0.9, neg=0.05, neu=0.05)
    absa.save_cache(frame, path=path)

    back = absa.load_cache(path)
    assert list(back.columns) == absa.CACHE_COLUMNS
    assert back["pair_hash"].tolist() == frame["pair_hash"].tolist()
    assert back["pos"].iloc[0] == pytest.approx(0.9)


def test_cache_save_merges_and_never_truncates(tmp_path):
    # The sentiment.py truncation bug in cache-shaped form: a second call must
    # not shrink the file to just its own rows.
    path = tmp_path / "absa_cache.parquet"
    first = _cache_frame([("Tesla beat expectations.", "Tesla")])
    absa.save_cache(first, path=path)

    second = _cache_frame([("Ford raised guidance.", "Ford")])
    absa.save_cache(pd.concat([absa.load_cache(path), second], ignore_index=True), path=path)

    back = absa.load_cache(path)
    assert len(back) == 2
    assert set(back["pair_hash"]) == set(first["pair_hash"]) | set(second["pair_hash"])


def test_score_sentence_table_preserves_unrelated_cache_entries(tmp_path):
    # End-to-end version of the same guarantee, through the caller that owns the
    # save. score_pairs() returns rows only for THIS call's pairs, so writing it
    # alone would drop everything else in the file.
    path = tmp_path / "absa_cache.parquet"
    stale = _cache_frame([("Some other sentence entirely.", "Ford")], pos=0.11)
    text = "Tesla delivered record numbers this quarter."
    fresh = _cache_frame([(text, "Tesla")], pos=0.7)
    absa.save_cache(pd.concat([stale, fresh], ignore_index=True), path=path)

    absa.score_sentence_table(
        _frame([_sentence_row(text, mentions_target=True)]), TICKER, cache_path=path
    )

    back = absa.load_cache(path)
    assert set(stale["pair_hash"]).issubset(set(back["pair_hash"]))
    assert len(back) == 2


def test_corrupt_cache_is_treated_as_empty(tmp_path):
    path = tmp_path / "absa_cache.parquet"
    path.write_bytes(b"not a parquet file")
    assert len(absa.load_cache(path)) == 0
    assert list(absa.load_cache(path).columns) == absa.CACHE_COLUMNS


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_absa_aggregates_are_nan_when_score_columns_absent():
    rows = [
        _sentence_row("Tesla delivered record numbers.", sent_idx=0, mentions_target=True),
        _sentence_row("Ford reported strong truck sales.", sent_idx=1),
        _sentence_row("Musk said the case would proceed.", sent_idx=2, mentions_ceo=True),
    ]
    df = _frame(rows)
    for col in ("pos", "neg", "neu"):
        df[col] = 0.5

    features = aggregate_article_features(df)

    assert len(features) == 1
    for col in ABSA_FEATURE_COLUMNS:
        assert col in features.columns
        assert pd.isna(features.loc[0, col])
    # FinBERT columns are untouched.
    assert features.loc[0, "sent_entity_pos"] == pytest.approx(0.5)


def test_absa_aggregates_read_the_same_buckets_as_finbert():
    rows = [
        _sentence_row("Tesla delivered record numbers.", sent_idx=0, mentions_target=True),
        _sentence_row(
            "Build-A-Bear outperformed Tesla.",
            sent_idx=1,
            mentions_target=True,
        ),
        _sentence_row("Ford reported strong truck sales.", sent_idx=2),
        _sentence_row("Musk said the case would proceed.", sent_idx=3, mentions_ceo=True),
    ]
    df = _frame(rows)
    for col in ("pos", "neg", "neu"):
        df[col] = 0.5
    df["absa_pos"] = [0.8, 0.1, 0.4, 0.6]
    df["absa_neg"] = [0.1, 0.8, 0.4, 0.2]
    df["absa_neu"] = [0.1, 0.1, 0.2, 0.2]

    f = aggregate_article_features(df).iloc[0]

    assert f["absa_entity_pos"] == pytest.approx((0.8 + 0.1) / 2)
    assert f["absa_ceo_pos"] == pytest.approx(0.6)
    # Empty bucket -> NaN, never 0.
    ceo_less = df[df["sent_idx"] != 3]
    assert pd.isna(aggregate_article_features(ceo_less).iloc[0]["absa_ceo_pos"])


@pytest.mark.slow
def test_absa_model_reads_the_aspect_not_the_sentence():
    # The motivating case, against the real model: one sentence, two aspects,
    # opposite readings. Skipped if the backend is unavailable -- ABSA is
    # best-effort everywhere else in this module too.
    if not absa.is_available():
        pytest.skip("ABSA backend unavailable")
    text = "Build-A-Bear outperformed Tesla this quarter."
    scored = absa.score_pairs([(text, "Tesla"), (text, "Build-A-Bear")])
    by_hash = scored.set_index("pair_hash")
    tesla = by_hash.loc[absa.pair_hash(text, "Tesla")]
    bbw = by_hash.loc[absa.pair_hash(text, "Build-A-Bear")]
    assert tesla["neg"] > tesla["pos"]
    assert bbw["pos"] > tesla["pos"]


# ---------------------------------------------------------------------------
# Substitution grammar: the injected name is inflected to the surface it replaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,start,end,expected",
    [
        # Possessive pronoun -> possessive name.
        ("Its revenue increased 16% year over year.", 0, 3,
         "Tesla's revenue increased 16% year over year."),
        ("This is a direct result of its price cuts.", 27, 30,
         "This is a direct result of Tesla's price cuts."),
        ("Their factories ran below capacity all quarter.", 0, 5,
         "Tesla's factories ran below capacity all quarter."),
        # A surface that is already possessive keeps exactly one "'s".
        ("The company's margins slipped again this quarter.", 0, 13,
         "Tesla's margins slipped again this quarter."),
        # Subject+copula contractions.
        ("It's expanding the Berlin plant again this year.", 0, 4,
         "Tesla is expanding the Berlin plant again this year."),
        ("They're expanding the Berlin plant again this year.", 0, 7,
         "Tesla are expanding the Berlin plant again this year."),
        # Everything else: the bare name, the original behaviour.
        ("The company also raised prices across the lineup.", 0, 11,
         "Tesla also raised prices across the lineup."),
        ("It expects deliveries to keep climbing next year.", 0, 2,
         "Tesla expects deliveries to keep climbing next year."),
    ],
)
def test_build_pairs_inflects_the_injected_name(text, start, end, expected):
    out = absa.build_pairs(
        _frame(
            [
                _sentence_row(
                    text,
                    mentions_target=True,
                    resolved_by_coref=True,
                    mention_char_start=start,
                    mention_char_end=end,
                )
            ]
        ),
        TICKER,
    )
    assert out.loc[0, "absa_text"] == expected
    assert out.loc[0, "absa_aspect"] == "Tesla"
    # The published sentence is never rewritten.
    assert out.loc[0, "text"] == text


def test_build_pairs_never_substitutes_over_a_personal_possessive():
    """"his"/"her" are possessive pronouns, but they are PERSON pronouns.

    This case used to live in the inflection table above, asserting that
    "Analysts questioned his margin guidance" became "... questioned Tesla's
    margin guidance". That expectation was wrong, and it is the same defect the
    person-mention rule already forbids wearing a possessive costume: a company
    does not have a "his", so a coreference chain that links one to a company is
    in error, and injecting the company name over it speaks for a human being.
    is_substitutable_mention() therefore excludes the person pronouns from
    _MENTION_PRONOUNS, and the row is scored with its text UNCHANGED.

    _inflect_name() still maps "his" -> "Tesla's" in isolation (see the test
    below); that is correct as an inflection rule and simply unreachable for
    person pronouns now, which is the point.
    """
    text = "Analysts questioned his margin guidance for the year."
    out = absa.build_pairs(
        _frame(
            [
                _sentence_row(
                    text,
                    mentions_target=True,
                    resolved_by_coref=True,
                    mention_char_start=20,
                    mention_char_end=23,
                )
            ]
        ),
        TICKER,
    )
    assert out.loc[0, "absa_text"] == text
    assert out.loc[0, "absa_aspect"] == "Tesla"


# ---------------------------------------------------------------------------
# Never split a contraction: a bare-pronoun span followed by a clitic must
# consume the clitic and expand it, rather than leaving it stitched onto the
# injected name ("It's" -> span covers only "It" -> "Tesla's also profitable",
# which reads as a possessive instead of "Tesla IS also profitable").
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,start,end,expected",
    [
        # "'s" is the COPULA here, not a possessive, precisely because the span
        # is the bare pronoun "It", not "Its".
        ("It's also profitable this quarter.", 0, 2,
         "Tesla is also profitable this quarter."),
        ("They're also profitable this quarter.", 0, 4,
         "Tesla is also profitable this quarter."),
        ("It'll do well next quarter.", 0, 2,
         "Tesla will do well next quarter."),
        ("They've already invested heavily.", 0, 4,
         "Tesla has already invested heavily."),
        ("It'd be a surprise if it missed.", 0, 2,
         "Tesla would be a surprise if it missed."),
    ],
)
def test_build_pairs_expands_contraction_instead_of_splitting_it(text, start, end, expected):
    out = absa.build_pairs(
        _frame(
            [
                _sentence_row(
                    text,
                    mentions_target=True,
                    resolved_by_coref=True,
                    mention_char_start=start,
                    mention_char_end=end,
                )
            ]
        ),
        TICKER,
    )
    assert out.loc[0, "absa_text"] == expected
    assert out.loc[0, "absa_aspect"] == "Tesla"
    assert out.loc[0, "text"] == text


def test_substitute_expands_clitic_directly():
    # Unit-level check on _substitute() itself, isolated from build_pairs()'s
    # row plumbing.
    text = "It's also profitable"
    new_text, ok = absa._substitute(text, 0, 2, "Tesla")
    assert ok is True
    assert new_text == "Tesla is also profitable"
    # NOT the buggy "Tesla's also profitable" -- that reads as a possessive.
    assert new_text != "Tesla's also profitable"


def test_substitute_does_not_expand_a_genuinely_possessive_surface():
    # A possessive surface ("Its") is not a bare pronoun, so it must keep
    # _inflect_name()'s existing "<Name>'s" behaviour untouched -- there is no
    # clitic to consume because the span already covers the whole possessive
    # form and nothing follows it that looks like a clitic on a bare pronoun.
    text = "Its revenue increased"
    new_text, ok = absa._substitute(text, 0, 3, "Tesla")
    assert ok is True
    assert new_text == "Tesla's revenue increased"


def test_inflect_name_preserves_leading_capitalisation():
    # A capitalised surface (sentence-initial) yields a capitalised injection,
    # even when the display name itself is lowercase.
    assert absa._inflect_name("tesla", "Its") == "Tesla's"
    assert absa._inflect_name("tesla", "its") == "tesla's"
    assert absa._inflect_name("Tesla", "it's") == "Tesla is"
    assert absa._inflect_name("Tesla", "they're") == "Tesla are"
    assert absa._inflect_name("Tesla", "the company") == "Tesla"


# ---------------------------------------------------------------------------
# A person mention is never substituted with the company name
# ---------------------------------------------------------------------------


def test_build_pairs_never_substitutes_over_a_person_mention():
    # Article 139905684: the coref chain reached this sentence through "Mr
    # Musk", and substituting produced "It also claims Tesla had himself
    # suggested ...". A person mention is never spoken for by the company, so
    # the sentence is scored UNCHANGED.
    text = "It also claims the Tesla CEO had himself suggested turning OpenAI into a venture."
    start = text.index("the Tesla CEO")
    out = absa.build_pairs(
        _frame(
            [
                _sentence_row(
                    text,
                    mentions_target=True,
                    resolved_by_coref=True,
                    mention_char_start=start,
                    mention_char_end=start + len("the Tesla CEO"),
                )
            ]
        ),
        TICKER,
    )
    assert out.loc[0, "absa_text"] == text
    assert out.loc[0, "absa_aspect"] == "Tesla"


def test_build_pairs_resolved_row_with_no_span_scores_unchanged_text():
    # entity_filter records no span when every coref mention in the sentence is
    # a PERSON mention. That is "do not substitute", not a data-quality event.
    text = "It also claims Mr Musk had himself suggested a commercial venture."
    out = absa.build_pairs(
        _frame([_sentence_row(text, mentions_target=True, resolved_by_coref=True)]), TICKER
    )
    assert out.loc[0, "absa_text"] == text
    assert out.loc[0, "absa_aspect"] == "Tesla"


# ---------------------------------------------------------------------------
# A non-substitutable mention (entity_filter.is_substitutable_mention() is
# False) is never substituted over, even when a span was recorded for it --
# defence-in-depth against a bad span sneaking past entity_filter's own
# span-selection guard (see entity_filter._coref_hits_in_sentence()).
# ---------------------------------------------------------------------------


def test_substitute_resolved_returns_unchanged_text_for_non_referring_span():
    # "Brazil" is a proper noun a coref error dragged into the chain, not a
    # company mention -- entity_filter.is_substitutable_mention("Brazil") is
    # False, so absa.py must refuse to inject over it even though the span
    # itself is perfectly well-formed.
    text = "That list is Australia, Brazil, India, Korea, and Vietnam."
    start = text.index("Brazil")
    end = start + len("Brazil")

    new_text, outcome = absa._substitute_resolved(text, start, end, "Tesla")

    assert outcome == "not_mention"
    assert new_text == text
    # Byte-identical, not merely equal-looking.
    assert new_text is text or new_text.encode() == text.encode()


def test_build_pairs_non_referring_span_leaves_text_unchanged():
    # Real corpus breakage: "Both stocks trade at steep valuations" became
    # "Tesla trade at steep valuations" because the plural "stocks" was
    # substituted over. A row flagged resolved whose span covers a
    # non-referring surface must get absa_text == text (unchanged) while still
    # getting the correct absa_aspect, exactly like the "person" and "no_span"
    # outcomes.
    text = "Both stocks trade at steep valuations this quarter."
    start = text.index("Both stocks")
    end = start + len("Both stocks")
    out = absa.build_pairs(
        _frame(
            [
                _sentence_row(
                    text,
                    mentions_target=True,
                    resolved_by_coref=True,
                    mention_char_start=start,
                    mention_char_end=end,
                )
            ]
        ),
        TICKER,
    )
    assert out.loc[0, "absa_text"] == text
    assert out.loc[0, "absa_aspect"] == "Tesla"


