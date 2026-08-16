# Fusion sentence-labelling protocol

The instruction sheet for labelling the 300-sentence fusion evaluation set
(`references/fusion-labels.csv` holds the first annotator's labels). This
formalises the procedure notebook 2.4 section 3 describes in prose, so that a
second annotator can follow exactly the same one.

Read this in full before labelling the first row.

## The question

For every row you are given a sentence and an aspect company (`absa_aspect`,
in practice always Tesla). Answer:

> Reading only this sentence, how does it read **for the aspect company,
> financially**?

This is **not** "is this sentence positive". A sentence can be cheerful and
still be bad news for the company you are judging, and vice versa. Judge the
company named in `absa_aspect`, never the sentence as a whole, and never some
other company the sentence happens to mention.

## The scale

Fill in `label` with exactly one of these five values:

| label | meaning |
|---|---|
| `+1.0` | clearly good news for the company |
| `+0.5` | mildly good |
| `0.0`  | no sentiment toward the company — neutral, purely factual, boilerplate, or about somebody else |
| `-0.5` | mildly bad |
| `-1.0` | clearly bad news for the company |

Five points rather than three because the evaluation needs to test whether a
model ranks **strength** correctly, not only direction. A three-way
positive/negative/neutral scheme could only test direction.

### Choosing between `+/-1.0` and `+/-0.5`

Use the full point for material, concrete financial news: earnings, deliveries,
margins, guidance, price targets, analyst rating changes, large share moves,
market-share shifts. Use the half point for softer signals: sentiment, opinion,
speculation, incremental product news, mild framing.

### When to use `0.0`

`0.0` is a real answer and should be used freely. It covers:

- purely factual or descriptive sentences ("Tesla designs and sells electric
  vehicles");
- sentences whose subject is another company, where the aspect company is
  merely named in passing;
- navigation furniture, promotional interruptions, disclaimers, truncated
  fragments;
- sentences where the direction genuinely cannot be recovered from the sentence
  alone.

Roughly a third of a typical sample lands here. Do not stretch for a signed
label when the sentence does not support one.

## Worked examples

These come from the real corpus and each illustrates a trap.

**"BYD capitalized on Tesla's weakness."** → **−1.0** toward Tesla.
The event is good for BYD, but "weakness" attaches to Tesla. Do not let a
sentence's overall positive tone bleed onto the aspect it is not about.

**"Rivian's $19bn market cap pales in comparison to Tesla's $1.2 trillion
valuation."** → **+0.5** toward Tesla, despite the word "pales".
"Pales" describes Rivian. The sentence is favourable to Tesla. Check which
company a negative word grammatically attaches to before labelling off it.

**"The ETF is up 28% YTD, though it has cooled on Tesla weakness."** → **−0.5**
toward Tesla, despite the sentence describing a rising fund.
The clause about the aspect is the relevant one. The ETF's good year is
evidence about the ETF.

**"Tesla designs and sells high-performance electric vehicles alongside its
expanding energy generation and storage product lines."** → **0.0**.
Descriptive. No sentiment either way.

**"Its net income tumbled further, plunging 61% in the quarter and 46% for the
year, or a $3.3 billion decline."** → **−1.0**.
Concrete, material, unambiguous.

## Rules

1. **Sentence alone.** You are shown the sentence without its article, by
   design. Do not try to reconstruct the surrounding context. If a pronoun's
   referent is unrecoverable, that is what `0.0` is for.
2. **Financial reading.** The question is how this reads for the company's
   business or its stock, not whether the sentence is pleasant. "Musk is a
   superhuman" is mild praise; a 61% earnings decline is material.
3. **Comparatives: judge the aspect only.** When two companies are compared,
   ignore entirely how the sentence reads about the other one.
4. **Do not consult any model output.** Do not open the pipeline's scores, the
   fusion module, notebook 2.4, or any existing label file while labelling.
   An annotator shown a model's answer tends to agree with it, and the
   resulting agreement figure then measures conformity rather than
   correctness.
5. **Label every row.** There is no skip option; `0.0` covers the
   "nothing here" case.

## Output

A CSV with exactly two columns, `sample_id` and `label`, one row per input
sentence, `label` being one of the five values above written as a number.

## A note on what a second annotator is for

The first annotator also designed the scoring variants being evaluated, which
is a conflict the blind procedure reduces but does not remove. A second set of
labels serves two purposes: it estimates how **subjective** this task is (if
two careful annotators disagree often, then a model scoring 0.9 against either
set is near the ceiling and small differences between models are noise), and it
tests whether the first annotator's labels follow from **this document** rather
than from their own expectations.

If the second annotator is an LLM rather than a person, the second purpose
holds and the first does not: agreement then measures whether the protocol is
specified tightly enough to be applied consistently, and must never be reported
as human validation.
