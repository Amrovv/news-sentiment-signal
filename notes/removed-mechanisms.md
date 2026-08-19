# Removed mechanisms — what was built, what went, and why

**Written:** 2026-08-19.

Notebooks 2.0, 2.1 and 2.3 describe a pipeline that has since been cut down twice. They were never
reconciled with the cuts, so a reader working forward through the series meets several features that
no longer exist and one whose reasoning survives **only in a git commit message** — the weakest
possible place to keep it.

This file is the archive. No code, no re-derivation: what the mechanism did, what killed it, and what
would have to be true to bring it back.

Nothing here is a live part of the pipeline. For what *is* live, see `ARCHITECTURE.md`.

---

## 1. The other-company family

**Removed in commit `b1fcc6f`, "Remove other-company detection; ABSA supersedes what it was for".**

Columns deleted — sentence schema 16 → 12: `mentions_other`, `is_comparative`, `other_source`,
`other_key`. Article schema 58 → 44: `sent_other_mean_*`, `absa_other_mean_*`, `n_other_sents`,
`n_comparative_sents`, and **both `excl_comp` families**. Test suite 248 → 223.

**What it was for.** It stopped sentences naming *other* companies from contaminating the target's
sentiment. That was necessary while FinBERT was the only scorer: FinBERT scores a whole sentence and
structurally cannot tell whose sentiment it is reading. So a rival's good news, sitting in a Tesla
article, scored as Tesla's good news.

**Why it went.** ABSA scores sentiment *toward an aspect*. It is handed the target explicitly, so it
gets comparative and multi-entity sentences right by construction. The justification for the whole
family was superseded rather than merely outgrown.

**The measurements that decided it:**

- Of 10,502 NER-sourced non-boilerplate rows, only 1,369 also carried `mentions_target`. The other
  **9,133 existed purely to feed an unvalidated feature family**.
- Removing the detector **unblocked coreference on 663 sentences** it had been pre-empting, 344 of
  which coref resolves to the target — so the detector was costing real signal, not just neutral.
- The registry roster powering the coref ambiguity guard was worth **269 target sentences (1.8%)**.
  Read by hand, most are genuinely not Tesla: xAI, ETFs, other automakers, transcript noise.
- `excl_comp` was **forced out rather than chosen** — it is defined against `is_comparative`, so it
  could not survive that column.

**Two things deliberately NOT done, both live footguns:**

1. **spaCy's `ner` component stays ENABLED.** PERSON entities are load-bearing in three places
   (`_is_person_like()`, the coref person-chain guard, `mentions_ceo`). Disabling `ner` to "clean up
   after" the ORG detector would break all three **silently**. `_get_nlp()` now says so in a comment.
2. **`COMPANIES` was NOT emptied.** Deleting non-target entries raises `KeyError` in
   `_build_ticker_patterns()` for any other ticker, which would break the NVDA transfer and KO control
   runs. It survives as a registry of *potential targets*; what went is the code path that read
   non-target entries as *rivals*.

**What would justify bringing it back:** a scorer worse at attribution than ABSA, or evidence that ABSA
mis-attributes on a class of sentence the old detector caught. Neither exists today. Note that notebook
2.7 §4–§5 — the 68-distinct-referent error tail — is decisive evidence *against* a curated confusor
roster.

---

## 2. The spaCy ORG NER other-company detector

Removed in the same commit. Precision was **50–65%**, and `NER_ORG_STOPLIST` went with it.

Its remaining traces are three explanatory comments in `entity_filter.py` and the untracked
`company_registry.py`, which references it as history. Notebook 2.3 §1.2 already carries a SUPERSEDED
banner.

**See §1's point 1: `ner` itself stays on.** This is the single most likely mistake a future cleanup
will make.

---

## 3. The gold set

**Removed in commit `4d85bc0`, "Remove the gold set; 2.6 answered its question better".**
Deleted: `stock_predictor/text/goldset.py`, `references/goldset-*`, `tests/test_goldset.py`. A grep for
`goldset` across `stock_predictor/` and `tests/` returns nothing.

**Why it went, and this is the part worth keeping:** the design was flawed, not merely redundant. The
sampler presented sentences **without their surrounding context**, then asked a labeller to judge
whether the sentiment attribution was right — a judgement that frequently *requires* the preceding
sentences. It was abandoned unlabelled rather than labelled and discarded.

Notebook 2.6 then answered the same question properly, and its own §2–§3 record two further invalidated
attempts at the same trap. The transferable lesson, from 2.6: **when an evaluation says almost
everything is broken, suspect the evaluation first.**

---

## 4. The sourced Finnhub company registry

**Rejected in the removal record — "measured net harmful".** The concrete failure: it tagged Tesla's own
Optimus robot as a rival.

**Status: shelved, not deleted, and still an open decision.** `stock_predictor/text/company_registry.py`
(522 lines) and `data/external/company_registry.parquet` (22k rows) are both **untracked**. The module
is imported by nothing, has no test file, and is referenced only from notebook 2.7.

It builds an alias table from a whole-exchange listing (Finnhub or SEC EDGAR), with a corpus-screening
pass (`_CorpusScreen`) that decides which bare names are safe to match case-insensitively.

**This decision is recorded here, not made here.** Adopt it (needs an importer and tests) or delete it
— but do so deliberately. The measured verdict against it is "net harmful", and notebook 2.7 §4–§5
independently argues against curated confusor rosters.

---

## 5. The anaphora recency heuristic

**Not removed — retained dormant behind `USE_ANAPHORA_FALLBACK = False`** (commit `e524dc8`).
`resolve_anaphora()` still exists and is still tested. It contributes **0 rows** corpus-wide, verified.

**Why it was turned off.** Notebook 2.6's audit, on equivalent samples: the heuristic scored **13/100**,
coreference **74/100**. The passage was determinate in 99% of cases, so this is not labeller
uncertainty. **61% of the heuristic's failures resolved to a different named company** — it was not
merely missing, it was confidently wrong, which is worse.

**Two numbers disagree on its cost** and neither notebook explains the gap:
- Notebook 2.6's closing table: cost **313** target sentences (2.1%).
- `config.py`'s `USE_ANAPHORA_FALLBACK` comment: **278** non-boilerplate target sentences (1.9%).

Most likely boilerplate inclusion, but that is inference, not evidence. Treat both as approximate.

**Collateral, and this matters:** two mechanisms are live, tested code that is **unreachable** while the
flag is off — the question-sentence antecedent rule (2.1 §2) and scoped descriptor resolution
(`_resolve_descriptor()`, 2.3 §1.3). They are not dead code, but they are inert. Do not measure them
without turning the flag on first, and do not delete them as unused.

**What would justify re-enabling:** evidence that coref leaves a class of anaphor the heuristic caught.
The 2.6 audit found the reverse.

---

## 6. Maverick coreference — two withdrawn recommendations

Notebook 2.7 §8, §8b, §8c. Both recommendations that came out of this programme were later withdrawn by
the same notebook, which is why nothing shipped.

1. **Stage 4's ensemble-disagreement gate on span rows** — recommended in §8.5, **withdrawn in §11.5**
   once the Stage 1 judge measured better on the same rows. No `mv_state` or `maverick` symbol exists
   anywhere in `stock_predictor/` or `tests/`.
2. **Maverick as a replacement backend** — decided against in §8c.4: *"do not swap the backend now."*
   `COREF_MODEL` is unchanged.
3. **`mv_state` as a downstream feature** — recommended twice, never built. The only artifact is
   `data/interim/maverick_agreement.parquet`.

**Worth preserving from this programme:** §8c's *pre-registered decision rule* and its resolution. It is
the only record that "genuinely better backend" and "merely more conservative backend" were separated
rather than conflated — the trap §8b was written to avoid.

---

## 7. Smaller removals

| mechanism | fate | note |
|---|---|---|
| Bare `"it"` in `GENERIC_ANAPHORA` | **DEAD** — Wave 1b | Major precision hole; matched anywhere. Replaced by `_SENT_INITIAL_IT_RE`. `GENERIC_ANAPHORA` is now `["the company", "the firm"]`. |
| First-mention-wins antecedent rule | **SUPERSEDED** — Wave 4 | Replaced by subject-position (`SUBJECT_DEPS`, `_match_is_subject()`). |
| Tiered aliases `unambiguous`/`person`/`anaphoric` | **SUPERSEDED** by `config.COMPANIES` | `ALIASES` / `OTHER_COMPANIES` survive in `config.py` as derived shims, kept *only* so notebook 2.0 stays re-runnable — which it is not (see §8). |
| `score_headlines` cache truncation | **DEAD (fixed)** | Both save paths now merge. Confirmed 2,101 → 23,619 cache entries. |
| `gated` fusion variant | **DEAD as a candidate, LIVE as a column** | Notebook 2.4 §9: identical to `fin` in every stratum, and structurally cannot flip a sign. Not in `AGGREGATED_VARIANTS`. **Someone will propose this again.** |
| `sign_graft` deadband | **SUPERSEDED** by confidence weighting | `DEADBAND = 0.0`. Sweep table copied verbatim into `fusion.py`. |
| `mean_blend` "carried alongside" | **NEVER BUILT** | Notebook 2.4 recommends carrying it; 2.5 §1.1 explicitly drops it. **2.4 was never amended — a direct contradiction between the two notebooks.** 2.5 won. |

---

## 8. Consequences for the notebooks

Notebooks **2.0, 2.1 and 2.3 cannot be re-executed** against the current package — they read columns
deleted in `b1fcc6f`. The cells that read them are marked in place.

Notebook **2.4 cannot be re-executed either**, and this one had teeth: its 300-row sample is defined by
strata built from `is_comparative`, and its 900 hand labels are keyed by `sample_id` alone. The sample
has now been **frozen to `references/fusion-sample-frozen.csv`** — reconstructed from the git blob at
commit `94228ec` and verified against all four label files (300/300/300/30). See
`notes/stage4-scripts/freeze_fusion_sample.py`.

**Do not redraw that sample against the current schema.** It would silently invalidate all 900 labels.
