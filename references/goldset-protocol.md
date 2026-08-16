# Gold-set labelling protocol

This document is the instruction sheet for a human annotator labelling
`references/goldset-sample.csv` (or whatever path
`goldset.write_annotation_sheet()` was pointed at; notebook 2.3 section 9.1
writes that one). The withheld model outputs live alongside it in
`references/goldset-predictions.csv` — do not open that file until labelling is
finished; see "Do not consult the model" below. Read this document in full
before labelling the first row.

## The question

For every row, read the `text` column and answer:

> Reading only this sentence, is the sentiment expressed TOWARD `absa_aspect`
> positive, negative, or neutral?

This is **not** "is this sentence positive." A sentence can be a clearly
positive sentence while being negative, neutral, or completely silent about
the company you are asked to judge. `absa_aspect` names the company whose
sentiment you are rating -- always judge that company, never the sentence as a
whole, never any other company the sentence happens to mention.

## Label vocabulary

Fill in the `label` column with exactly one of:

- `positive`
- `negative`
- `neutral`
- `unclear`

`unclear` is reserved for sentences where the aspect company's role genuinely
cannot be determined from the sentence alone -- not for sentences that are
merely hard to score. Do not fold an `unclear` case into `neutral` to avoid
using the word; they mean different things downstream. `unclear` rows are
reported and analyzed SEPARATELY from the accuracy computation and are
EXCLUDED from the accuracy number itself -- a model cannot be scored right or
wrong against a label that says the true answer is unknowable from the text
given.

## Confidence

Fill in the `confidence` column with exactly one of:

- `high`
- `low`

Use `low` whenever you had to guess, whenever two readings of the sentence
seemed equally plausible, or whenever you would not be surprised to change
your mind on a second read.

## Notes

`notes` is free text, optional. Use it for anything that would help someone
auditing your labels later: an ambiguity you resolved a particular way, a
sentence that reads oddly out of context, a typo in the source text, etc.

## Worked examples

**"BYD capitalized on Tesla's weakness."**
Label toward Tesla: **negative**. The event described (capitalizing on
weakness) is positive for BYD, but the word "weakness" attaches directly to
Tesla. The positive framing belongs to the other company; don't let a
sentence's overall positive tone bleed onto the aspect it isn't about.

**"Rivian's $19bn market cap pales in comparison to Tesla's $1.2 trillion
valuation."**
Label toward Tesla: **positive**, despite the word "pales." "Pales" describes
Rivian's position, not Tesla's -- Tesla is the company with the enormous
valuation in this sentence, which is a favorable comparison for Tesla. Do not
label off surface-negative words without checking which company they
grammatically attach to.

**"The ETF is up 28% YTD, though it has cooled on Tesla weakness."**
Label toward Tesla: **negative**, despite the sentence overall describing a
rising ETF. The clause about Tesla specifically ("cooled on Tesla weakness")
is the part relevant to the aspect, and it is negative. The ETF's good year is
not evidence about Tesla; it is evidence about the ETF.

## Comparative sentences

When `stratum` is `comparative`, the sentence names both the target company
and at least one other company (e.g. "X outperformed Tesla this quarter").
Label sentiment toward the ASPECT COMPANY ONLY (`absa_aspect`). Ignore how the
sentence reads about the other named company entirely -- a sentence that
sounds very positive about the other company can still be strongly negative
toward the aspect company, and vice versa. This is the single most important
rule in this protocol: comparative sentences are in the gold set specifically
because sentence-level sentiment models get them backwards (see the first
worked example above).

## Anaphora- and coref-resolved sentences

When `stratum` is `anaphora_resolved` or `coref_resolved`, you are being shown
the sentence WITHOUT the surrounding article that let the pipeline resolve
what "the company" or "it" referred to. This is deliberate: labelling on the
sentence alone tests whether the sentence, once the pipeline believes it knows
the referent, actually carries recoverable sentiment about that referent.

If you cannot tell from the sentence alone who or what "it"/"the company"/"the
automaker" refers to, label `unclear` rather than guessing. Do not try to
reconstruct the missing context or assume the aspect named in `absa_aspect` is
correct -- your job is to judge whether the SENTENCE ITSELF, read cold,
supports a sentiment judgment about that aspect, not to independently verify
the pipeline's resolution.

A high `unclear` rate in these two strata specifically is not a labelling
failure -- it is itself a finding about how much of the pipeline's anaphora/
coref resolution work is happening on sentences that carry no self-contained
sentiment signal at all. Report the `unclear` rate per stratum; do not smooth
it away.

## Do not consult the model

Do not look up, run, or in any way consult the pipeline's own FinBERT or ABSA
outputs while labelling. Those columns are deliberately withheld from the
annotation sheet for exactly this reason: an annotator shown a model's answer
tends to agree with it, and the resulting accuracy figure then measures
agreement with the model rather than correctness. If you already know how the
model scored a particular sentence (e.g. from earlier notebook work), label it
as if you did not know.
