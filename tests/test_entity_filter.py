import pandas as pd
import pytest

from stock_predictor.config import MIN_SENT_CHARS
from stock_predictor.text import coref
from stock_predictor.text import entity_filter as _entity_filter
from stock_predictor.text.entity_filter import (
    flag_boilerplate,
    map_coref_clusters,
    process_articles,
    resolve_anaphora,
    split_sentences,
    tag_sentences,
)

TICKER = "TSLA"

# A small multi-company corpus reused by the coref tests. Each body mixes an
# explicit mention, an anaphoric follow-up, and an incidental rival mention --
# i.e. the shape the heuristic is known to get wrong.
COREF_CORPUS = pd.DataFrame(
    {
        "article_id": [101, 102, 103],
        "processed_body": [
            "Tesla delivered record numbers this quarter. "
            "The company also raised prices across its lineup. "
            "It expects deliveries to keep climbing next year.",
            "Lucid Motors went public this week. "
            "Lucid's market debut was led by Peter Rawlinson, Tesla's former engineer. "
            "The company then delivered its first sedans to customers.",
            "Ford reported strong truck sales in the quarter. "
            "The automaker also raised its dividend for shareholders. "
            "Tesla shares fell after the report was published.",
        ],
    }
)


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
    assert row["resolved_by_anaphora"] == False


def test_subject_beats_first_mention():
    # Dependency-based antecedent selection, replacing first-mention-wins. Ford
    # is named FIRST but sits in a prepositional phrase ("According to Ford");
    # Tesla is the grammatical subject. The sentence is about Tesla, so the
    # antecedent must be the target -- verified through the following
    # "The company..." sentence, which is what the antecedent actually feeds.
    sentences = [
        "According to Ford, Tesla is expanding its Berlin plant.",
        "The company said the expansion would finish next year.",
    ]
    df = resolve_anaphora(30, sentences, TICKER, use_anaphora_fallback=True)

    # Ford is no longer detectable (other-company detection removed), so the
    # only antecedent candidate is the target -- which is what the following
    # "The company..." sentence must inherit.
    assert df.iloc[1]["mentions_target"] == True
    assert df.iloc[1]["resolved_by_anaphora"] == True


def test_target_is_subject_flag():
    subj = tag_sentences(31, ["Tesla raised prices."], TICKER)
    assert subj.iloc[0]["target_is_subject"] == True
    assert subj.iloc[0]["mentions_target"] == True

    # Incidental appositive mention: the target is named but is not what the
    # sentence is about.
    non_subj = tag_sentences(
        31, ["Lucid's debut was led by Peter Rawlinson, Tesla's former engineer."], TICKER
    )
    assert non_subj.iloc[0]["mentions_target"] == True
    assert non_subj.iloc[0]["target_is_subject"] == False


def test_descriptor_decays_with_gap():
    # Descriptor resolution obeys the same ANAPHORA_MAX_GAP decay as generic
    # anaphora: a qualifying company named too far back is not carried forward.
    from stock_predictor.config import ANAPHORA_MAX_GAP

    filler = [
        f"This is unrelated filler sentence number {i} about the weather."
        for i in range(ANAPHORA_MAX_GAP)
    ]
    sentences = ["Ford reported strong truck sales.", *filler, "The automaker cut prices."]
    df = resolve_anaphora(37, sentences, TICKER, use_anaphora_fallback=True)

    last = df.iloc[-1]
    assert last["mentions_target"] == False
    assert last["resolved_by_anaphora"] == False


def test_question_sentences_do_not_update_antecedent():
    # Promo/rhetorical questions name companies without establishing a topic.
    sentences = [
        "Ford reported strong truck sales.",
        "Missed Nvidia and Tesla?",
        "The company also raised its dividend.",
    ]
    df = resolve_anaphora(22, sentences, TICKER, use_anaphora_fallback=True)

    # The question's own flag still fires.
    assert df.iloc[1]["mentions_target"] == True

    # But it must not become the antecedent: the following "The company..."
    # sentence has nothing to inherit and stays untagged. (Before other-company
    # detection was removed this asserted the antecedent was still Ford; the
    # rule under test -- a question never anchors -- is the same either way.)
    assert df.iloc[2]["mentions_target"] == False
    assert df.iloc[2]["resolved_by_anaphora"] == False


def test_sentence_initial_it_resolves_but_midsentence_it_does_not():
    # Bare "it" was demoted from GENERIC_ANAPHORA to a case-sensitive
    # sentence-initial pattern: "It also raised..." is genuinely anaphoric,
    # "analysts said it was hard to believe" is not.
    sentences = [
        "Tesla posted record deliveries.",
        "It also raised guidance for the year.",
        "Analysts said it was hard to believe the results.",
    ]
    df = resolve_anaphora(23, sentences, TICKER, use_anaphora_fallback=True)

    assert df.iloc[1]["mentions_target"] == True
    assert df.iloc[1]["resolved_by_anaphora"] == True

    assert df.iloc[2]["mentions_target"] == False
    assert df.iloc[2]["resolved_by_anaphora"] == False


def test_derived_compat_views_match_registry():
    # ALIASES / OTHER_COMPANIES are deprecated views derived from COMPANIES,
    # kept only so notebook 2.0 stays re-runnable.
    from stock_predictor.config import ALIASES, COMPANIES, OTHER_COMPANIES

    assert ALIASES["TSLA"]["unambiguous"] == COMPANIES["TSLA"]["names"]
    assert ALIASES["TSLA"]["person"] == COMPANIES["TSLA"]["person"]
    assert ALIASES["TSLA"]["anaphoric"] == COMPANIES["TSLA"]["descriptors"]

    # Every registry entry is present in both views (no entry is omitted).
    assert set(ALIASES) == set(COMPANIES)
    assert set(OTHER_COMPANIES) == set(COMPANIES)
    for key, entry in COMPANIES.items():
        assert OTHER_COMPANIES[key] == entry.get("names", [])
        assert ALIASES[key]["unambiguous"] == entry.get("names", [])
        assert ALIASES[key]["person"] == entry.get("person", [])
        assert ALIASES[key]["anaphoric"] == entry.get("descriptors", [])


def test_flag_boilerplate():
    # Exact sentence text repeated across >= BOILERPLATE_MIN_ARTICLES distinct
    # articles is corpus boilerplate; repeated across fewer is not.
    boiler = "Image source: Getty Images."
    rare = "Tesla delivered record numbers this quarter."
    rows = [{"article_id": i, "text": boiler} for i in range(5)]
    rows += [{"article_id": i, "text": rare} for i in range(2)]
    df = pd.DataFrame(rows)

    out = flag_boilerplate(df, min_articles=5)
    assert out[out["text"] == boiler]["is_boilerplate"].all()
    assert not out[out["text"] == rare]["is_boilerplate"].any()
    # Input frame untouched.
    assert "is_boilerplate" not in df.columns


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


def test_anaphora_decays_after_max_gap():
    # Long-article regression: an explicit mention many sentences back should
    # stop being attributed to generic anaphora once ANAPHORA_MAX_GAP is
    # exceeded, instead of drifting onto unrelated later content (the
    # dominant error mode found in the sentences.parquet hand-check --
    # see notebooks/a_entity_sentiment.ipynb section 4.3).
    from stock_predictor.config import ANAPHORA_MAX_GAP

    filler = [f"This is unrelated filler sentence number {i} about the weather." for i in range(ANAPHORA_MAX_GAP - 1)]
    sentences = ["Tesla delivered record numbers this quarter.", *filler, "The company said demand was strong."]
    df = resolve_anaphora(6, sentences, TICKER, use_anaphora_fallback=True)

    # Sentence just inside the gap (gap == ANAPHORA_MAX_GAP) still resolves.
    assert df.iloc[-1]["mentions_target"] == True
    assert df.iloc[-1]["resolved_by_anaphora"] == True

    # One sentence further out, the antecedent has decayed.
    sentences_too_far = sentences + ["The firm reported strong results."]
    df2 = resolve_anaphora(6, sentences_too_far, TICKER, use_anaphora_fallback=True)
    last = df2.iloc[-1]
    assert last["mentions_target"] == False
    assert last["resolved_by_anaphora"] == False


# ---------------------------------------------------------------------------
# USE_ANAPHORA_FALLBACK: the recency heuristic defaults OFF.
#
# A context-aware audit of 100 anaphora-resolved sentences (auditor saw a
# 6-sentence window, judged determinate in 99% of rows) found the heuristic
# correct on only 13 of 100 -- coreference scored 74% on the equivalent
# sample -- so the default leaves those sentences untagged rather than
# guessing. The capability stays behind the flag rather than being deleted.
# ---------------------------------------------------------------------------


def test_anaphora_fallback_default_leaves_13pct_precision_heuristic_off():
    sentences = [
        "Tesla delivered record numbers this quarter.",
        "The company also raised prices.",
    ]
    # Default: config.USE_ANAPHORA_FALLBACK (False) -- no keyword passed.
    df = resolve_anaphora(70, sentences, TICKER)

    row = df.iloc[1]
    assert row["mentions_target"] == False
    assert row["resolved_by_anaphora"] == False
    assert pd.isna(row["anaphor_char_start"])
    assert pd.isna(row["anaphor_char_end"])

    # With the flag explicitly re-enabled, the old behaviour returns.
    df_on = resolve_anaphora(70, sentences, TICKER, use_anaphora_fallback=True)
    row_on = df_on.iloc[1]
    assert row_on["mentions_target"] == True
    assert row_on["resolved_by_anaphora"] == True


def test_anaphora_fallback_off_does_not_regress_explicit_or_coref_paths():
    # Precedence 1 (explicit) and precedence 2 (coref) must tag normally with
    # the fallback OFF -- this is the thing that must not regress when the
    # heuristic (precedence 3) is disabled by default.
    text = "Tesla delivered record numbers. The company also raised prices."
    spans = split_sentences([text], return_spans=True)[0]
    assert len(spans) == 2

    # Explicit mention: unaffected by the flag either way.
    explicit_off = tag_sentences(71, spans, TICKER, use_anaphora_fallback=False)
    assert explicit_off.iloc[0]["mentions_target"] == True
    assert explicit_off.iloc[0]["resolved_by_anaphora"] == False

    # Coref-resolved sentence: still tags normally with the heuristic OFF,
    # since coref never falls through to the gated block at all.
    coref_start = text.index("The company")
    mentions = [(0, 5, "TARGET"), (coref_start, coref_start + len("The company"), "TARGET")]
    coref_off = tag_sentences(72, spans, TICKER, coref_mentions=mentions, use_anaphora_fallback=False)
    assert coref_off.iloc[1]["mentions_target"] == True
    assert coref_off.iloc[1]["resolved_by_coref"] == True
    assert coref_off.iloc[1]["resolved_by_anaphora"] == False


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


# ---------------------------------------------------------------------------
# Coreference resolution (stock_predictor/text/coref.py)
#
# Tests that need the real coref model are marked slow, matching the convention
# in test_sentiment.py. Everything else must run with coref DISABLED and must
# reproduce the pre-coref behaviour exactly -- the heuristic is the fallback and
# stays the tested default.
# ---------------------------------------------------------------------------


def test_coref_unavailable_falls_back_to_heuristic(monkeypatch):
    # Graceful degradation: with the backend reporting itself unavailable,
    # process_articles must still run to completion and produce exactly the
    # heuristic's tags (no exception, no coref-resolved rows). The heuristic
    # itself defaults OFF (config.USE_ANAPHORA_FALLBACK), so this test -- which
    # is specifically about the fallback mechanism -- must opt back into it
    # explicitly to observe resolved_by_anaphora firing at all.
    monkeypatch.setattr(coref, "is_available", lambda: False)

    out = process_articles(COREF_CORPUS, TICKER, use_coref=True, use_anaphora_fallback=True)
    baseline = process_articles(COREF_CORPUS, TICKER, use_coref=False, use_anaphora_fallback=True)

    assert not out["resolved_by_coref"].any()
    assert out["resolved_by_anaphora"].any()
    pd.testing.assert_frame_equal(out, baseline)


def test_coref_mentions_only_fill_unnamed_sentences():
    # PRECEDENCE, without needing the model: a sentence with an explicit alias
    # match keeps its explicit tag and is NOT marked coref-resolved, even when a
    # coref mention covers it. Only the un-named sentence takes the coref tag.
    text = "Tesla delivered record numbers. The company also raised prices."
    spans = split_sentences([text], return_spans=True)[0]
    assert len(spans) == 2
    # Hand-built cluster: the explicit "Tesla" plus the anaphor "The company".
    mentions = [(0, 5, "TARGET"), (text.index("The company"), text.index("The company") + 11, "TARGET")]

    df = tag_sentences(50, spans, TICKER, coref_mentions=mentions)

    assert df.iloc[0]["mentions_target"] == True
    assert df.iloc[0]["resolved_by_coref"] == False
    assert df.iloc[1]["mentions_target"] == True
    assert df.iloc[1]["resolved_by_coref"] == True
    assert df.iloc[1]["resolved_by_anaphora"] == False
    assert df.iloc[1]["text"][
        int(df.iloc[1]["anaphor_char_start"]) : int(df.iloc[1]["anaphor_char_end"])
    ] == "The company"


def test_anaphor_spans_are_within_sentence():
    # Both mechanisms record the resolved mention's span relative to the row's
    # own text. Heuristic path here (no model needed); the coref path is covered
    # by the slow test below. The heuristic is opted into explicitly since it
    # defaults off (config.USE_ANAPHORA_FALLBACK).
    out = process_articles(COREF_CORPUS, TICKER, use_coref=False, use_anaphora_fallback=True)
    resolved = out[out["anaphor_char_start"].notna()]
    assert len(resolved) > 0

    for _, row in resolved.iterrows():
        start, end = int(row["anaphor_char_start"]), int(row["anaphor_char_end"])
        assert 0 <= start < end <= len(row["text"])
        mention = row["text"][start:end]
        assert mention
        assert mention == mention.strip()


def test_span_columns_are_nullable_int():
    # Schema stability: the columns must be Int64 whether or not anything was
    # resolved, so downstream reads do not see object/float64 by accident.
    out = process_articles(COREF_CORPUS, TICKER, use_coref=False)
    assert str(out["anaphor_char_start"].dtype) == "Int64"
    assert str(out["anaphor_char_end"].dtype) == "Int64"
    assert out["anaphor_char_start"].isna().any()


def test_needs_score_unaffected_by_new_columns():
    # sentiment.needs_score() reads only mentions_target/ceo and
    # is_boilerplate; adding coref columns must not disturb it.
    from stock_predictor.text.sentiment import needs_score

    out = process_articles(COREF_CORPUS, TICKER, use_coref=False)
    mask = needs_score(out)
    assert len(mask) == len(out)
    expected = (
        out["mentions_target"] | out["mentions_ceo"]
    ) & ~out["is_boilerplate"]
    assert (mask == expected).all()


@pytest.mark.slow
def test_coref_backend_smoke():
    # The backend loads and returns char-offset clusters into the ORIGINAL text.
    assert coref.is_available()
    text = "Tesla delivered record numbers this quarter. The company also raised prices."
    clusters = coref.resolve_documents([text])[0]
    assert clusters
    surfaces = {text[s:e] for cluster in clusters for s, e in cluster}
    assert "Tesla" in surfaces


@pytest.mark.slow
def test_explicit_mention_beats_coref():
    out = process_articles(COREF_CORPUS, TICKER, use_coref=True)
    explicit = out[out["text"].str.contains("Tesla")]
    assert len(explicit) > 0
    assert not explicit["resolved_by_coref"].any()


@pytest.mark.slow
def test_coref_and_heuristic_flags_mutually_exclusive():
    out = process_articles(COREF_CORPUS, TICKER, use_coref=True)
    assert int((out["resolved_by_coref"] & out["resolved_by_anaphora"]).sum()) == 0


def test_person_chain_does_not_key_to_target():
    # A PERSON coreference chain routinely contains an appositive mention whose
    # surface embeds a company name ("Tesla CEO Elon Musk"). Keying the chain on
    # that mention made every "he"/"Musk" sentence in the document a target
    # sentence -- the article 140122179 bug (a Musk-vs-Altman lawsuit story).
    # mentions_target has never been settable by a person-tier match, and coref
    # must not be a back door into it.
    text = (
        "Tesla CEO Elon Musk sued Sam Altman this week. "
        "He said the case would proceed. "
        "Kalshi's similar market has Musk winning at 55%."
    )
    appositive = (0, len("Tesla CEO Elon Musk"))
    he = (text.index("He said"), text.index("He said") + 2)
    musk = (text.rindex("Musk"), text.rindex("Musk") + 4)
    cluster = [[appositive, he, musk]]
    # The PERSON entity spaCy finds inside the appositive mention.
    person_spans = [(text.index("Elon Musk"), text.index("Elon Musk") + len("Elon Musk"))]

    mentions, stats = map_coref_clusters(cluster, text, TICKER, person_spans=person_spans)

    assert mentions == []
    assert stats["resolved"] == 0
    assert stats["unresolved"] == 1

    # Control for the PERSON-overlap rule specifically. The appositive above is
    # now ALSO blocked by the title-NP rule ("CEO"), so to isolate the
    # person-span mechanism the control drops the job title from the surface:
    # without person_spans, "Tesla's Elon Musk" still keys the chain to TARGET
    # (the pre-fix behaviour, and the default for callers with no parse).
    untitled = (
        "Tesla's Elon Musk sued Sam Altman this week. "
        "He said the case would proceed. "
        "Kalshi's similar market has Musk winning at 55%."
    )
    u_cluster = [
        [
            (0, len("Tesla's Elon Musk")),
            (untitled.index("He said"), untitled.index("He said") + 2),
            (untitled.rindex("Musk"), untitled.rindex("Musk") + 4),
        ]
    ]
    unfiltered, unfiltered_stats = map_coref_clusters(u_cluster, untitled, TICKER)
    assert unfiltered_stats["resolved"] == 1
    assert all(m[2] == "TARGET" for m in unfiltered)


def test_person_span_still_tagged_when_cluster_keyed_by_other_mention():
    # The filter changes only the KEYING decision. A person-like mention stays a
    # member of its cluster: when some other, non-person mention names the
    # company, the chain resolves and every mention -- the person-like one
    # included -- still tags its sentence.
    text = (
        "Tesla said deliveries rose. "
        "Tesla CEO Elon Musk called the quarter strong. "
        "He expects more growth."
    )
    plain = (0, 5)
    appositive = (
        text.index("Tesla CEO Elon Musk"),
        text.index("Tesla CEO Elon Musk") + len("Tesla CEO Elon Musk"),
    )
    he = (text.index("He expects"), text.index("He expects") + 2)
    cluster = [[plain, appositive, he]]
    person_spans = [(text.index("Elon Musk"), text.index("Elon Musk") + len("Elon Musk"))]

    mentions, stats = map_coref_clusters(cluster, text, TICKER, person_spans=person_spans)

    assert stats["resolved"] == 1
    assert {m[2] for m in mentions} == {"TARGET"}
    # All three mentions are emitted, including the person-like appositive.
    assert {(m[0], m[1]) for m in mentions} == {plain, appositive, he}


def test_title_noun_phrase_does_not_key_to_target():
    # The PERSON-overlap rule only catches mentions that literally contain a
    # person's name. "the Tesla CEO" / "The Tesla Inc. (NASDAQ: TSLA) CEO" are
    # appositive TITLE noun phrases naming a PERSON via their company: they hold
    # no PERSON entity at all, so person_spans cannot see them, yet keying the
    # person-chain to the company is exactly what the person-tier design forbids
    # (mentions_target is never settable by a person-tier match alone).
    # Real cases: article 140122179 sent_idx 19, article 141052996 sent_idx 8/26.
    text = (
        "The Tesla CEO sued Sam Altman this week. "
        "He said the case would proceed. "
        "Kalshi's similar market has him winning at 55%."
    )
    title_np = (0, len("The Tesla CEO"))
    he = (text.index("He said"), text.index("He said") + 2)
    him = (text.index("has him") + 4, text.index("has him") + 7)
    cluster = [[title_np, he, him]]

    # No person_spans at all: the title token alone must block the keying.
    mentions, stats = map_coref_clusters(cluster, text, TICKER)

    assert mentions == []
    assert stats["resolved"] == 0
    assert stats["unresolved"] == 1

    # The longer real-world form, with the ticker furniture spelled out.
    long_text = "The Tesla Inc. (NASDAQ: TSLA) CEO sold shares again."
    long_cluster = [[(0, len("The Tesla Inc. (NASDAQ: TSLA) CEO"))]]
    long_mentions, long_stats = map_coref_clusters(long_cluster, long_text, TICKER)
    assert long_mentions == []
    assert long_stats["resolved"] == 0

    # Keying-only, exactly like the person-span rule: when another mention in the
    # same chain names the company plainly, the chain resolves and the title NP
    # is still emitted and still tags its sentence.
    keyed = "Tesla said deliveries rose. The Tesla CEO called the quarter strong."
    plain = (0, 5)
    title = (keyed.index("The Tesla CEO"), keyed.index("The Tesla CEO") + len("The Tesla CEO"))
    keyed_mentions, keyed_stats = map_coref_clusters([[plain, title]], keyed, TICKER)
    assert keyed_stats["resolved"] == 1
    assert {(m[0], m[1]) for m in keyed_mentions} == {plain, title}
    assert {m[2] for m in keyed_mentions} == {"TARGET"}


# ---------------------------------------------------------------------------
# Coreference on-disk cache (coref.load_cache / save_cache / resolve_documents)
#
# These use a fake model so they stay fast and can COUNT inference calls, which
# is the whole point: a warm run must do zero.
# ---------------------------------------------------------------------------


class _FakePrediction:
    def __init__(self, clusters):
        self._clusters = clusters

    def get_clusters(self, as_strings=False):
        return self._clusters


class _SpyModel:
    """Stands in for the fastcoref model and counts what it was asked to do."""

    def __init__(self):
        self.calls = 0
        self.texts_seen = []

    def predict(self, texts, max_tokens_in_batch=None):
        self.calls += 1
        self.texts_seen.extend(texts)
        return [_FakePrediction([[(0, 5), (len(t) - 5, len(t))]]) for t in texts]


@pytest.fixture
def spy_model(monkeypatch):
    spy = _SpyModel()
    monkeypatch.setattr(coref, "_load_model", lambda: spy)
    return spy


def test_coref_cache_round_trip(tmp_path, spy_model):
    cache_path = tmp_path / "coref_cache.parquet"
    texts = [
        "Tesla delivered record numbers this quarter and the company raised prices.",
        "Ford reported strong truck sales and the automaker raised its dividend.",
    ]

    first = coref.resolve_documents(texts, cache_path=cache_path)
    assert spy_model.calls == 1
    assert len(spy_model.texts_seen) == 2
    assert cache_path.exists()

    second = coref.resolve_documents(texts, cache_path=cache_path)
    # Warm run: zero model inference, identical clusters.
    assert spy_model.calls == 1
    assert len(spy_model.texts_seen) == 2
    assert second == first
    assert all(clusters for clusters in second)


def test_coref_cache_merges_not_replaces(tmp_path, spy_model):
    # sentiment.py had a truncation bug from replacing the cache instead of
    # merging it; the same shape of bug here would throw away a full corpus run.
    cache_path = tmp_path / "coref_cache.parquet"
    seeded = pd.DataFrame(
        {"text_hash": ["deadbeefdeadbeef"], "clusters_json": ["[[[0, 5]]]"]}
    )
    coref.save_cache(seeded, path=cache_path)

    coref.resolve_documents(["A completely unrelated Tesla document."], cache_path=cache_path)

    on_disk = pd.read_parquet(cache_path)
    assert "deadbeefdeadbeef" in set(on_disk["text_hash"])
    assert len(on_disk) == 2


def test_coref_corrupt_cache_is_treated_as_empty(tmp_path, spy_model):
    cache_path = tmp_path / "coref_cache.parquet"
    cache_path.write_bytes(b"this is not a parquet file at all")

    # Neither the loader nor a full resolve may raise on a corrupt cache.
    assert len(coref.load_cache(path=cache_path)) == 0

    out = coref.resolve_documents(["Tesla delivered record numbers."], cache_path=cache_path)
    assert len(out) == 1
    assert spy_model.calls == 1


@pytest.mark.slow
def test_coref_anaphor_spans_are_within_sentence():
    out = process_articles(COREF_CORPUS, TICKER, use_coref=True)
    rows = out[out["resolved_by_coref"] & out["anaphor_char_start"].notna()]
    for _, row in rows.iterrows():
        start, end = int(row["anaphor_char_start"]), int(row["anaphor_char_end"])
        assert 0 <= start < end <= len(row["text"])
        mention = row["text"][start:end]
        assert mention and mention == mention.strip()


# ---------------------------------------------------------------------------
# Expletive "It" (dummy subject) is not an anaphor
# ---------------------------------------------------------------------------


def test_expletive_it_is_not_resolved_by_the_sentence_initial_rule():
    # "It's no coincidence that ..." has a DUMMY subject: the "It" refers to
    # nothing, so the sentence is about no company at all. Before this, the
    # sentence-initial "It" rule handed it to the last-named company and the
    # ABSA substitution turned it into "Tesla's no coincidence that ...".
    text = (
        "Tesla delivered record numbers this quarter. "
        "It's no coincidence that both of these models have starting prices "
        "under $50,000."
    )
    spans = split_sentences([text], return_spans=True)[0]
    # use_anaphora_fallback=True: this test is specifically about the
    # sentence-initial "It" rule inside the heuristic, which defaults off
    # (config.USE_ANAPHORA_FALLBACK) and must be opted into to observe it.
    df = tag_sentences(60, spans, TICKER, use_anaphora_fallback=True)

    assert df.iloc[0]["mentions_target"] == True
    row = df.iloc[1]
    assert row["mentions_target"] == False
    assert row["resolved_by_anaphora"] == False
    assert pd.isna(row["anaphor_char_start"])

    # Control: a genuinely referential sentence-initial "It" in the same
    # position still resolves.
    ok = (
        "Tesla delivered record numbers this quarter. "
        "It expects deliveries to keep climbing next year."
    )
    ok_df = tag_sentences(
        61, split_sentences([ok], return_spans=True)[0], TICKER, use_anaphora_fallback=True
    )
    assert ok_df.iloc[1]["mentions_target"] == True
    assert ok_df.iloc[1]["resolved_by_anaphora"] == True


def test_expletive_coref_mention_sets_no_tag_and_no_span():
    # Coref links the dummy "It" into the Tesla chain. The mention must be
    # dropped outright: no key, no span, nothing for ABSA to substitute over.
    text = (
        "Tesla delivered record numbers this quarter. "
        "It is important to note that both models cost under $50,000."
    )
    spans = split_sentences([text], return_spans=True)[0]
    assert len(spans) == 2
    it_start = text.index("It is important")
    mentions = [(0, 5, "TARGET"), (it_start, it_start + 2, "TARGET")]

    df = tag_sentences(62, spans, TICKER, coref_mentions=mentions)

    assert df.iloc[1]["mentions_target"] == False
    assert df.iloc[1]["resolved_by_coref"] == False
    assert pd.isna(df.iloc[1]["anaphor_char_start"])


def test_expletive_detection_prefers_the_parse():
    # The parse is authoritative when it is available: an extraposed
    # "It is <adj/NP> that/to ..." is expletive, a plain referential "It" is not,
    # and the surface regex is not consulted for either.
    nlp = _entity_filter._get_nlp()
    expletive = "It's no coincidence that both of these models are cheap."
    referential = "It expects deliveries to keep climbing next year."
    assert _entity_filter._is_expletive_mention(nlp(expletive), 0, 2, expletive, 0) is True
    assert _entity_filter._is_expletive_mention(nlp(referential), 0, 2, referential, 0) is False


def test_expletive_regex_fallback_when_no_parse_is_available():
    # doc=None is the no-parse path (a caller with strings and a parser-less
    # pipeline). The conservative regex requires sentence-initial "It"/"It's",
    # a form of "be", a short phrase and an extraposed "that"/"to" clause.
    fire = [
        "It's no coincidence that both of these models are cheap.",
        "It is important to note that shares fell.",
        "It is clear that demand fell.",
    ]
    hold = [
        "It's a good quarter for the company.",
        "It expects deliveries to keep climbing next year.",
        "Its revenue increased 16% year over year.",
    ]
    for text in fire:
        assert _entity_filter._is_expletive_mention(None, 0, 2, text, 0) is True, text
    for text in hold:
        assert _entity_filter._is_expletive_mention(None, 0, 2, text, 0) is False, text
    # The pattern is anchored at the start of the sentence, so it says nothing
    # about a mid-sentence mention.
    mid = "Analysts said it is clear that demand fell."
    assert _entity_filter._is_expletive_mention(None, 15, 17, mid, 15) is False


def test_person_denoting_title_np_does_not_key_a_cluster():
    # Article 139905684: a 36-mention Musk chain keyed to TARGET off the single
    # mention "the Tesla billionaire's" -- the same class of leak as "the Tesla
    # CEO", with a person-denoting head noun instead of a job title.
    text = (
        "The Tesla billionaire's lawyers filed the motion. "
        "He said the case would proceed."
    )
    cluster = [[(0, len("The Tesla billionaire's")), (text.index("He said"), text.index("He said") + 2)]]
    mentions, stats = map_coref_clusters(cluster, text, TICKER)
    assert mentions == []
    assert stats["unresolved"] == 1


# ---------------------------------------------------------------------------
# is_substitutable_anaphor: the span selected for ABSA injection must be a
# recognisable company anaphor, never an arbitrary piece of a coref chain.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "surface",
    [
        "it", "its", "they", "their", "them", "theirs",
        "the company", "the company's", "the company’s",  # curly apostrophe
        "company", "the stock", "the EV maker", "this company",
        "the electric vehicle (ev) maker", "The Company", "the automaker's",
    ],
)
def test_is_substitutable_anaphor_allows_real_company_anaphors(surface):
    assert _entity_filter.is_substitutable_anaphor(surface) is True


@pytest.mark.parametrize(
    "surface",
    [
        # PERSON pronouns -- simply absent from the closed class, deliberately.
        "he", "she", "I", "you", "his company",
        # First-person plural: removed from _ANAPHORIC_PRONOUNS. In news prose
        # a "we"/"our" is almost always inside a quote from a PERSON, not the
        # company speaking in its own voice -- a context-aware audit found 10
        # corpus substitutions produced subject-verb disagreement this way
        # ("'We want the future to look like the future,' Musk said" became
        # "Tesla want the future to look like the future").
        "we", "our", "us", "ours", "we're", "we’re",
        # A proper noun a coref error dragged into the chain.
        "Brazil", "Spotify",
        # A product, not the company itself (see the separate PRODUCT_KEYS
        # test below for the registry-driven version of this).
        "robotaxi",
        # A product name shaped like a company NP.
        "the Model Y",
        # PLURAL noun phrases denote MORE than one company.
        "both companies", "the companies", "these companies",
        # A generic noun with no company sense.
        "the shares", "the earnings call", "the vehicle",
        # A possessive clitic / bare demonstrative with no head noun at all.
        "'s", "this", "that",
        # An arbitrary long NP that merely ends in a company-ish word's
        # possessive modifier -- real breakage: "...whether FSD would stop for
        # a school bus with its stop arm extended..." -> "...stop for Tesla.".
        "a school bus with its stop arm extended and red lights flashing",
        "",
    ],
)
def test_is_substitutable_anaphor_rejects_non_company_surfaces(surface):
    assert _entity_filter.is_substitutable_anaphor(surface) is False


def test_is_substitutable_anaphor_rejects_none():
    assert _entity_filter.is_substitutable_anaphor(None) is False


def test_is_substitutable_anaphor_rejects_first_person_plural():
    # First-person plural is no longer treated as company anaphora -- see
    # _ANAPHORIC_PRONOUNS. In news prose "we"/"our" is almost always inside a
    # quote from a PERSON, not the company speaking in its own voice.
    assert _entity_filter.is_substitutable_anaphor("we") is False
    assert _entity_filter.is_substitutable_anaphor("our") is False
    assert _entity_filter.is_substitutable_anaphor("us") is False
    assert _entity_filter.is_substitutable_anaphor("ours") is False
    assert _entity_filter.is_substitutable_anaphor("we're") is False


def test_is_substitutable_anaphor_rejects_registry_product_names():
    # PRODUCT_KEYS is the union of every COMPANIES entry's "products" tier
    # (see entity_filter._build_product_keys()). A product name is never the
    # company itself, no matter whose product it is, so every one of them must
    # be rejected -- none of the current registry's product names happen to be
    # shaped like a bare company head noun, so on the live registry these are
    # all caught by the head-noun check upstream of the PRODUCT_KEYS test.
    for product_key in _entity_filter.PRODUCT_KEYS:
        assert _entity_filter.is_substitutable_anaphor(product_key) is False
    assert "fsd" in _entity_filter.PRODUCT_KEYS
    assert "robotaxi" in _entity_filter.PRODUCT_KEYS


def test_is_substitutable_anaphor_product_key_check_itself(monkeypatch):
    # Isolate the PRODUCT_KEYS branch specifically: a surface that WOULD
    # otherwise pass every other test (a determiner + a valid company head
    # noun) must still be rejected when that exact phrase is itself a
    # registered product name. No current registry entry happens to overlap a
    # company head noun this way, so this test manufactures one rather than
    # leaving the branch unexercised.
    monkeypatch.setattr(_entity_filter, "PRODUCT_KEYS", frozenset({"the stock"}))
    assert _entity_filter.is_substitutable_anaphor("the stock") is False
    # An unrelated determiner+head-noun phrase is unaffected.
    assert _entity_filter.is_substitutable_anaphor("the company") is True


# ---------------------------------------------------------------------------
# _coref_hits_in_sentence: span SELECTION must skip non-substitutable mentions
# ---------------------------------------------------------------------------


def test_coref_hits_prefers_a_later_pronoun_over_an_earlier_junk_mention():
    # A sentence containing both a junk (non-anaphoric) surface EARLY and a
    # good pronoun LATER must select the PRONOUN's span, not the junk one --
    # strictly better than filtering only after the fact in absa.py, since the
    # earlier mechanism would otherwise have already thrown the good pronoun
    # away in favour of position.
    text = "That list is Australia, Brazil, India, Korea, and Vietnam, and it led the region."
    brazil = (text.index("Brazil"), text.index("Brazil") + len("Brazil"))
    it = (text.index(" it led") + 1, text.index(" it led") + 3)
    mentions = [(brazil[0], brazil[1], "TARGET"), (it[0], it[1], "TARGET")]

    keys, best = _entity_filter._coref_hits_in_sentence(
        mentions, offset=0, length=len(text), doc=None, text=text, person_spans=[]
    )

    assert keys == {"TARGET"}
    assert best == it


def test_coref_hits_returns_key_with_no_span_when_only_mention_is_junk():
    # The sentence is still resolved (a real chain named the company), but
    # nothing in it is fit to be substituted over -- the span stays None and
    # absa.py scores the sentence unsubstituted rather than injecting over
    # "Brazil".
    text = "That list is Australia, Brazil, India, Korea, and Vietnam."
    brazil = (text.index("Brazil"), text.index("Brazil") + len("Brazil"))
    mentions = [(brazil[0], brazil[1], "TARGET")]

    keys, best = _entity_filter._coref_hits_in_sentence(
        mentions, offset=0, length=len(text), doc=None, text=text, person_spans=[]
    )

    assert keys == {"TARGET"}
    assert best is None
