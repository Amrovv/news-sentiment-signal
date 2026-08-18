"""Persist the 60 hand labels for the Maverick-only rows (Stage 4b closeout).

verdict: 'target' = the sentence genuinely refers to the target company,
         'other'  = it refers to something else.
borderline: defensible either way; counted as labelled but reported both ways.
Convention held constant with the existing 270-row eval set: a reference to a
PRODUCT rather than the company (Model S/X, FSD, Semi) counts as 'other', matching
the existing 'the Tesla Semi truck (product, not the company)' precedent.
"""
import pandas as pd

# row_id -> (verdict, referent, borderline, note)
LABELS = {
    # ---- span rows ----
    "MVONLY-2":   ("target", "the target company", False, "'its approach' = camera-only autonomy approach; sents 25-29 all Tesla"),
    "MVONLY-3":   ("target", "the target company", False, "'it received a wave of positive commentary' = Tesla"),
    "MVONLY-4":   ("target", "the target company", False, "'the stock' = TSLA, upgraded by Baird"),
    "MVONLY-20":  ("other", "the United Kingdom (market, not a company)", False, "'it does not have the high tariffs' = the UK"),
    "MVONLY-26":  ("target", "the target company", False, "'The stock is up about 10%' = TSLA"),
    "MVONLY-41":  ("other", "the Model S and Model X (products, not the company)", False, "'them' = the S and X cars being discontinued"),
    "MVONLY-42":  ("other", "the Model S and Model X (products, not the company)", False, "'their production' = the S and X cars"),
    "MVONLY-45":  ("target", "the target company", False, "'its reach' = Tesla's reach"),
    "MVONLY-50":  ("target", "the target company", True, "'They weren't able to offer those deals' - reads as Tesla vs prior sentence, but Lemonade is a live alternative"),
    "MVONLY-59":  ("other", "Li Auto", False, "'the company operated 542 retail stores' = Li Auto"),
    "MVONLY-70":  ("target", "the target company", False, "'the stock has seen a 1.7% decline' = TSLA"),
    "MVONLY-71":  ("other", "the article itself (Simply Wall St disclaimer)", False, "'It does not constitute a recommendation' = the article"),
    "MVONLY-73":  ("other", "Tesla FSD (product, not the company)", False, "'it was negatively impacted by a lack of marketing' = the FSD technology"),
    "MVONLY-80":  ("target", "the target company", False, "'They said they were going to spend $20 billion' = Tesla capex"),
    "MVONLY-84":  ("target", "the target company", False, "'they said they're going to be spending' = Tesla capex ramp"),
    "MVONLY-97":  ("other", "Tesla FSD (product, not the company)", False, "'its current form' = the FSD system Sweden wants blocked"),
    "MVONLY-102": ("target", "the target company", False, "'the EV stock' = TSLA"),
    "MVONLY-106": ("target", "the target company", False, "'the stock' = TSLA retail interest"),
    "MVONLY-112": ("target", "the target company", False, "'The stock has shed roughly a third of its value' = TSLA"),
    "MVONLY-118": ("target", "the target company", True, "'the automotive business' = Tesla's auto segment; segment-not-company, but internal to Tesla"),
    "MVONLY-119": ("target", "the target company", True, "'the car business' = Tesla's auto segment; same segment-vs-company call as MVONLY-118"),
    "MVONLY-122": ("target", "the target company", False, "'this is still, at its core, an automotive business' = Tesla (sent 146 names it)"),
    "MVONLY-123": ("target", "the target company", False, "'They've always had absurdly high margins' = Tesla"),
    "MVONLY-125": ("target", "the target company", False, "'They moved 35% more metal' = Tesla auto"),
    "MVONLY-128": ("target", "the target company", False, "'they underspent' = Tesla capex"),
    "MVONLY-129": ("target", "the target company", False, "'They had a chart in their earnings report' = Tesla"),
    "MVONLY-130": ("target", "the target company", False, "'They are expanding to more cities' = Tesla robotaxi"),
    "MVONLY-133": ("target", "the target company", False, "'they're thinking about the technology' = Tesla"),
    "MVONLY-134": ("target", "the target company", False, "'they're in seven cities now' = Tesla robotaxi"),
    "MVONLY-135": ("target", "the target company", False, "'they would be in seven cities' = Tesla robotaxi"),
    # ---- no-span rows ----
    "MVONLY-6":   ("other", "Lightship", False, "'we thought it was time to bring RVs up to date' = Lightship founders"),
    "MVONLY-7":   ("other", "Lightship", False, "RV/camping trend, Lightship interview"),
    "MVONLY-10":  ("other", "Lightship", False, "'we took a ground up approach' = Lightship"),
    "MVONLY-12":  ("other", "Lightship", False, "'This thing' = the Lightship RV"),
    "MVONLY-21":  ("other", "Chinese EV companies", False, "sentence is about Chinese EV makers competing in the UK"),
    "MVONLY-32":  ("target", "the target company", True, "Musk quote on being ousted; topic is Tesla voting control, but the sentence itself is a bare quote attribution"),
    "MVONLY-46":  ("target", "the target company", False, "analyst on trading S and X for Optimus = Tesla revenue mix"),
    "MVONLY-55":  ("other", "Li Auto / NIO", False, "'Also Read: NIO And Li Auto Fall' - related-link fragment"),
    "MVONLY-66":  ("other", "BYD", False, "'A lot of those are hybrids' = BYD's vehicles"),
    "MVONLY-67":  ("other", "Lamborghini", False, "CEO Stephan Winkelmann = Lamborghini"),
    "MVONLY-75":  ("target", "the target company", False, "'That implies roughly 15 million shares of daily retail volume' = TSLA volume"),
    "MVONLY-76":  ("target", "the target company", True, "'Now consider where that flow sits' - transitional, but the flow is TSLA's"),
    "MVONLY-90":  ("other", "generic order-flow analytics definition", False, "explains the Power Inflow signal in general, not Tesla"),
    "MVONLY-92":  ("other", "SpaceX / an X user's post", False, "about the SpaceX pay package"),
    "MVONLY-95":  ("other", "Rivian", False, "'Shares of Rivian have formed a Death Cross'"),
    "MVONLY-105": ("other", "California voucher / charging infrastructure", True, "about counties' charging capacity; Tesla Semi is the article's topic but not this sentence's subject"),
    "MVONLY-107": ("other", "generic debt-to-equity ratio definition", False, "methodology sentence, no company referent"),
    "MVONLY-110": ("other", "Michael Burry", False, "'The investor, best known for predicting the 2008 collapse' = Burry"),
    "MVONLY-114": ("target", "the target company", False, "'This is a company mid-transition' = Tesla"),
    "MVONLY-115": ("target", "the target company", False, "'Shares are down 31.24% year to date' = TSLA"),
    "MVONLY-116": ("target", "the target company", False, "'This is a Tesla-specific reset' - explicit"),
    "MVONLY-144": ("other", "Elon Musk (personal aside)", False, "'I'm a little under the weather here' - health aside on the earnings call"),
    "MVONLY-145": ("other", "Elon Musk (personal aside)", False, "continuation of the health aside"),
    "MVONLY-148": ("target", "the target company", False, "analyst asking Tesla about its own robotaxi rollout"),
    "MVONLY-150": ("target", "the target company", True, "'What do you want to see' - continuation of the regulator question to Tesla, contentless alone"),
    "MVONLY-152": ("other", "regulation in general", True, "Tesla exec's general view on good regulation; topic is policy, not Tesla"),
    "MVONLY-154": ("target", "the target company", False, "asking Tesla about third-party rideshare distribution for robotaxi"),
    "MVONLY-155": ("target", "the target company", False, "about Tesla's robotaxi fleet size per city"),
    "MVONLY-164": ("other", "Megapack batteries in SpaceX data centers", True, "Megapack is a Tesla product, but the sentence is about AI data-centre power draw"),
    "MVONLY-166": ("other", "SpaceX", False, "O'Leary on buying SpaceX shares"),
}

frame = pd.read_parquet("data/interim/stage4b_maverick_only_frame.parquet")
sample = frame[frame["sample_n60"]].copy()

missing = set(sample["row_id"]) - set(LABELS)
extra = set(LABELS) - set(sample["row_id"])
assert not missing, f"unlabelled rows: {sorted(missing)}"
assert not extra, f"labels for rows not in the sample: {sorted(extra)}"

sample["verdict"] = sample["row_id"].map(lambda r: LABELS[r][0])
sample["referent"] = sample["row_id"].map(lambda r: LABELS[r][1])
sample["borderline"] = sample["row_id"].map(lambda r: LABELS[r][2])
sample["note"] = sample["row_id"].map(lambda r: LABELS[r][3])
sample["has_span"] = sample["row_kind"] == "span"

out = "data/eval/mvonly_eval_labelled.parquet"
sample.to_parquet(out, index=False)
print(f"wrote {out}: {len(sample)} rows")
print(sample.groupby(["has_span", "verdict"]).size().to_string())
