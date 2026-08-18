"""Convention change: a reference to the TARGET's own product counts as `target`.

Previously a mention resolving to one of the company's own products (the Semi,
the Model S/X, FSD) was labelled `other`, on the reasoning that the pipeline
substitutes the COMPANY NAME over the anaphor and "ending their production" ->
"ending Tesla's production" is a different assertion from the original.

The project owner has ruled the other way, and for the downstream task that is
the better call: the feature being built is sentiment about the company for a
price model, and sentiment about a company's own products IS sentiment about the
company. A sentence saying the Semi is delayed is negative information about
Tesla, and discarding it loses real signal.

SCOPE. Only the TARGET's own products flip. A competitor's product stays `other`
("Rivian (R2 crossover)" is about Rivian), and a target product discussed in
another company's context stays `other` ("Megapack batteries in SpaceX data
centers" -- the sentence is about SpaceX's data centres, not about Tesla).
"""
import pandas as pd

FLIPS = {
    "data/eval/coref_eval_labelled.parquet": {
        "SPAN-16": "the target company (via its own product, the Semi truck)",
    },
    "data/eval/mvonly_eval_labelled.parquet": {
        "MVONLY-41": "the target company (via its own products, the Model S and Model X)",
        "MVONLY-42": "the target company (via its own products, the Model S and Model X)",
        "MVONLY-73": "the target company (via its own product, FSD)",
        "MVONLY-97": "the target company (via its own product, FSD)",
    },
}

NOTE = (
    "Referent is one of the target's OWN products. Relabelled target -> the "
    "convention now counts a company's own products as the company, since "
    "sentiment about them is sentiment about the company for the price model. "
    "Was 'other' before 2026-08-18."
)

for path, flips in FLIPS.items():
    df = pd.read_parquet(path)
    before = int((df["verdict"] == "other").sum())

    for row_id, referent in flips.items():
        mask = df["row_id"] == row_id
        assert mask.sum() == 1, f"{row_id} not found exactly once in {path}"
        assert df.loc[mask, "verdict"].iloc[0] == "other", f"{row_id} is already target"
        df.loc[mask, "verdict"] = "target"
        df.loc[mask, "referent"] = referent
        df.loc[mask, "note"] = NOTE

    df.to_parquet(path, index=False)
    after = int((df["verdict"] == "other").sum())
    print(f"{path}: errors {before} -> {after} ({len(flips)} flipped)")
    print(df.groupby(["has_span", "verdict"]).size().to_string())
    print()
