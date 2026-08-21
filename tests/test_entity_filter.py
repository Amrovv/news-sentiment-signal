import pandas as pd
import pytest

from stock_predictor.config import MIN_SENT_CHARS
from stock_predictor.text import coref
from stock_predictor.text import entity_filter as _entity_filter
from stock_predictor.text.entity_filter import (
    flag_boilerplate,
    map_coref_clusters,
    process_articles,
    split_sentences,
    tag_sentences,
)

TICKER = "TSLA"

# A small multi-company corpus reused by the coref tests. Each body mixes an
# explicit mention, a follow-up that only refers back to the company, and an
# incidental rival mention -- i.e. the shape that needs coreference to tag.
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
    assert row["resolved_by_coref"] == False


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


def test_mentions_ceo_independent_of_mentions_target():
    # Design decision (documented in entity_filter.tag_sentences docstring):
    # person-tier aliases are ambiguous (e.g. "Musk" could mean SpaceX/X), so
    # mentions_ceo is tracked independently and never sets mentions_target on
    # its own. A sentence with "Musk" but no unambiguous/referring Tesla
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


def test_coref_unavailable_tags_explicit_names_only(monkeypatch):
    # Graceful degradation: with the backend reporting itself unavailable,
    # process_articles must still run to completion -- no exception, and no
    # coref-resolved rows -- producing exactly the tags explicit alias matching
    # gives on its own.
    monkeypatch.setattr(coref, "is_available", lambda: False)

    out = process_articles(COREF_CORPUS, TICKER, use_coref=True)
    baseline = process_articles(COREF_CORPUS, TICKER, use_coref=False)

    assert not out["resolved_by_coref"].any()
    assert out["mentions_target"].any()
    pd.testing.assert_frame_equal(out, baseline)


def test_coref_mentions_only_fill_unnamed_sentences():
    # PRECEDENCE, without needing the model: a sentence with an explicit alias
    # match keeps its explicit tag and is NOT marked coref-resolved, even when a
    # coref mention covers it. Only the un-named sentence takes the coref tag.
    text = "Tesla delivered record numbers. The company also raised prices."
    spans = split_sentences([text], return_spans=True)[0]
    assert len(spans) == 2
    # Hand-built cluster: the explicit "Tesla" plus the mention "The company".
    mentions = [(0, 5, "TARGET"), (text.index("The company"), text.index("The company") + 11, "TARGET")]

    df = tag_sentences(50, spans, TICKER, coref_mentions=mentions)

    assert df.iloc[0]["mentions_target"] == True
    assert df.iloc[0]["resolved_by_coref"] == False
    assert df.iloc[1]["mentions_target"] == True
    assert df.iloc[1]["resolved_by_coref"] == True
    assert df.iloc[1]["text"][
        int(df.iloc[1]["mention_char_start"]) : int(df.iloc[1]["mention_char_end"])
    ] == "The company"


def test_mention_spans_are_within_sentence():
    # The resolved mention's span is recorded relative to the row's OWN text,
    # which is what absa.py slices to inject the company name. Hand-built
    # mentions here so no model is needed; the real backend is covered by the
    # slow test below.
    text = "Tesla delivered record numbers. The company also raised prices."
    spans = split_sentences([text], return_spans=True)[0]
    start = text.index("The company")
    mentions = [(0, 5, "TARGET"), (start, start + len("The company"), "TARGET")]

    out = tag_sentences(60, spans, TICKER, coref_mentions=mentions)
    resolved = out[out["mention_char_start"].notna()]
    assert len(resolved) > 0

    for _, row in resolved.iterrows():
        s_, e_ = int(row["mention_char_start"]), int(row["mention_char_end"])
        assert 0 <= s_ < e_ <= len(row["text"])
        mention = row["text"][s_:e_]
        assert mention
        assert mention == mention.strip()


def test_span_columns_are_nullable_int():
    # Schema stability: the columns must be Int64 whether or not anything was
    # resolved, so downstream reads do not see object/float64 by accident.
    out = process_articles(COREF_CORPUS, TICKER, use_coref=False)
    assert str(out["mention_char_start"].dtype) == "Int64"
    assert str(out["mention_char_end"].dtype) == "Int64"
    assert out["mention_char_start"].isna().any()


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
def test_coref_mention_spans_are_within_sentence():
    out = process_articles(COREF_CORPUS, TICKER, use_coref=True)
    rows = out[out["resolved_by_coref"] & out["mention_char_start"].notna()]
    for _, row in rows.iterrows():
        start, end = int(row["mention_char_start"]), int(row["mention_char_end"])
        assert 0 <= start < end <= len(row["text"])
        mention = row["text"][start:end]
        assert mention and mention == mention.strip()


# ---------------------------------------------------------------------------
# Expletive "It" (dummy subject) is not an mention
# ---------------------------------------------------------------------------


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
    assert pd.isna(df.iloc[1]["mention_char_start"])


def test_expletive_detection_reads_the_parse():
    # An extraposed "It is <adj/NP> that/to ..." is a dummy subject; a plain
    # referential "It" is not. The distinction is structural, so it is read off
    # the dependency parse rather than pattern-matched on the surface.
    nlp = _entity_filter._get_nlp()
    expletive = "It's no coincidence that both of these models are cheap."
    referential = "It expects deliveries to keep climbing next year."
    assert _entity_filter._is_expletive_mention(nlp(expletive), 0, 2) is True
    assert _entity_filter._is_expletive_mention(nlp(referential), 0, 2) is False


def test_expletive_detection_without_a_parse_is_not_an_expletive():
    # Fail-open on a doc with no dependency annotation: without the parse there
    # is no evidence of extraposition, and guessing from the surface is what the
    # removed regex fallback did. No parse means the mention is left alone.
    assert _entity_filter._is_expletive_mention(None, 0, 2) is False


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
# is_substitutable_mention: the span selected for ABSA injection must be a
# recognisable company mention, never an arbitrary piece of a coref chain.
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
def test_is_substitutable_mention_allows_real_company_mentions(surface):
    assert _entity_filter.is_substitutable_mention(surface) is True


@pytest.mark.parametrize(
    "surface",
    [
        # PERSON pronouns -- simply absent from the closed class, deliberately.
        "he", "she", "I", "you", "his company",
        # First-person plural: removed from _MENTION_PRONOUNS. In news prose
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
def test_is_substitutable_mention_rejects_non_company_surfaces(surface):
    assert _entity_filter.is_substitutable_mention(surface) is False


def test_is_substitutable_mention_rejects_none():
    assert _entity_filter.is_substitutable_mention(None) is False


def test_is_substitutable_mention_rejects_first_person_plural():
    # First-person plural is no longer treated as a company mention -- see
    # _MENTION_PRONOUNS. In news prose "we"/"our" is almost always inside a
    # quote from a PERSON, not the company speaking in its own voice.
    assert _entity_filter.is_substitutable_mention("we") is False
    assert _entity_filter.is_substitutable_mention("our") is False
    assert _entity_filter.is_substitutable_mention("us") is False
    assert _entity_filter.is_substitutable_mention("ours") is False
    assert _entity_filter.is_substitutable_mention("we're") is False


def test_is_substitutable_mention_rejects_registry_product_names():
    # PRODUCT_KEYS is the union of every COMPANIES entry's "products" tier
    # (see entity_filter._build_product_keys()). A product name is never the
    # company itself, no matter whose product it is, so every one of them must
    # be rejected -- none of the current registry's product names happen to be
    # shaped like a bare company head noun, so on the live registry these are
    # all caught by the head-noun check upstream of the PRODUCT_KEYS test.
    for product_key in _entity_filter.PRODUCT_KEYS:
        assert _entity_filter.is_substitutable_mention(product_key) is False
    assert "fsd" in _entity_filter.PRODUCT_KEYS
    assert "robotaxi" in _entity_filter.PRODUCT_KEYS


def test_is_substitutable_mention_product_key_check_itself(monkeypatch):
    # Isolate the PRODUCT_KEYS branch specifically: a surface that WOULD
    # otherwise pass every other test (a determiner + a valid company head
    # noun) must still be rejected when that exact phrase is itself a
    # registered product name. No current registry entry happens to overlap a
    # company head noun this way, so this test manufactures one rather than
    # leaving the branch unexercised.
    monkeypatch.setattr(_entity_filter, "PRODUCT_KEYS", frozenset({"the stock"}))
    assert _entity_filter.is_substitutable_mention("the stock") is False
    # An unrelated determiner+head-noun phrase is unaffected.
    assert _entity_filter.is_substitutable_mention("the company") is True


# ---------------------------------------------------------------------------
# _coref_hits_in_sentence: span SELECTION must skip non-substitutable mentions
# ---------------------------------------------------------------------------


def test_coref_hits_prefers_a_later_pronoun_over_an_earlier_junk_mention():
    # A sentence containing both a junk (non-referring) surface EARLY and a
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
