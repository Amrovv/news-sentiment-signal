"""Convention change #2: joint referents, fund holders and target-context
generics count as `target`.

RULE (project owner's call, 2026-08-18). A sentence counts as being about the
target when a reader would read it as a direct reference to the target, which
covers three classes previously labelled `other`:

  (a) JOINT/PLURAL referents that INCLUDE the target -- "both firms", "the two
      global EV leaders", "all three companies", "Musk's portfolio". The target
      is a named subject of the sentence, not a bystander, so the sentiment is
      partly about it.
  (b) FUNDS, BASKETS AND GROUPS THAT HOLD the target -- an ETF or index whose
      performance is being discussed and which holds the target as a
      constituent.
  (c) GENERIC DEFINITIONS AND THIRD-PARTY CAMPAIGNS raised in a target article
      -- a technical standard being defined, or an advocacy group's demands,
      where the article context makes a reader read it as characterising the
      target.

CARVE-OUT: INVERSE / SHORT instruments stay `other`. TSLQ is an inverse-Tesla
ETF: its sentiment moves OPPOSITE to the target, so accepting it does not dilute
the signal, it inverts it -- strictly worse than discarding the sentence. This is
the one case where "a reader could read it as a Tesla reference" and "it carries
Tesla's sentiment" point in opposite directions.

APPLIED TO THE WHOLE EVAL SET, not only to rows the judge accepted. Relabelling
just the rows a model happened to agree with would be scoring the labels against
the thing under test -- the circularity HANDOFF §9 warns about and notebook 2.6
documents twice.
"""
import pandas as pd

# (a) joint/plural referents including the target
JOINT = [
    "NOSPAN-51", "NOSPAN-54", "NOSPAN-62", "NOSPAN-99", "NOSPAN-100",
    "NOSPAN-102", "NOSPAN-108", "NOSPAN-115", "NOSPAN-121", "NOSPAN-126",
    "NOSPAN-139", "NOSPAN-163",
]
# (b) funds / baskets / groups holding the target. NOSPAN-21 (TSLQ) deliberately
# absent -- see the carve-out above.
HOLDER = [
    "NOSPAN-49", "NOSPAN-56", "NOSPAN-73", "NOSPAN-76", "NOSPAN-98",
    "NOSPAN-145", "NOSPAN-167",
]
# (c) generic definitions / third-party campaigns in a target article
GENERIC = ["NOSPAN-123", "NOSPAN-149"]

NOTE = (
    "Relabelled other -> target on 2026-08-18 under the reader-reference "
    "convention: a joint referent including the target, a fund holding the "
    "target, or a generic/third-party statement in a target article all read as "
    "direct references to the target. Inverse instruments (TSLQ) are excluded "
    "from this rule and remain errors, since their sentiment is sign-flipped."
)

PATH = "data/eval/coref_eval_labelled.parquet"
df = pd.read_parquet(PATH)
before = int((df["verdict"] == "other").sum())

for group, tag in [(JOINT, "joint referent incl. target"),
                   (HOLDER, "fund/basket holding the target"),
                   (GENERIC, "generic or third-party statement in a target article")]:
    for row_id in group:
        mask = df["row_id"] == row_id
        assert mask.sum() == 1, f"{row_id} not found exactly once"
        assert df.loc[mask, "verdict"].iloc[0] == "other", f"{row_id} already target"
        old = df.loc[mask, "referent"].iloc[0]
        df.loc[mask, "verdict"] = "target"
        df.loc[mask, "referent"] = f"the target company ({tag}; was: {old})"
        df.loc[mask, "note"] = NOTE
        # These rows are no longer defensible-either-way under the new rule --
        # the rule decides them -- so the borderline flag is cleared.
        df.loc[mask, "borderline"] = False

df.to_parquet(PATH, index=False)
after = int((df["verdict"] == "other").sum())
print(f"{PATH}: errors {before} -> {after} ({before - after} relabelled)")
print(df.groupby(["has_span", "verdict"]).size().to_string())
print(f"borderline remaining: {int(df['borderline'].sum())}")
