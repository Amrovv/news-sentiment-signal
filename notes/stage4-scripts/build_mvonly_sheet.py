"""Build the labelling context sheet for the 60 Maverick-only rows (Stage 4b).

Context window matches the convention used for every earlier labelling pass in this branch:
headline + 4 preceding sentences + the flagged sentence + 1 following sentence.
"""
import pandas as pd

frame = pd.read_parquet("data/interim/stage4b_maverick_only_frame.parquet")
sample = frame[frame["sample_n60"]].sort_values(["article_id", "sent_idx"]).reset_index(drop=True)

sentences = pd.read_parquet("data/sentences.parquet")
articles = pd.read_parquet("data/articles.parquet")

head_col = "headline" if "headline" in articles.columns else "title"
headlines = articles.set_index("article_id")[head_col].to_dict()

by_article = {aid: g.sort_values("sent_idx") for aid, g in sentences.groupby("article_id")}

lines = []
for _, row in sample.iterrows():
    aid, sidx = int(row["article_id"]), int(row["sent_idx"])
    g = by_article[aid]
    prev = g[(g["sent_idx"] < sidx) & (g["sent_idx"] >= sidx - 4)]
    nxt = g[(g["sent_idx"] > sidx) & (g["sent_idx"] <= sidx + 1)]

    lines.append(f"===== {row['row_id']} | article {aid} | sent {sidx} | {row['row_kind']} =====")
    lines.append(f"HEADLINE: {headlines.get(aid, '(missing)')}")
    if pd.notna(row["mv_span"]):
        text = str(row["text"])
        start = int(row["mv_span"])
        lines.append(f"MAVERICK ANAPHOR at char {start}: {text[start:start + 40]!r}")
    lines.append("")
    for _, p in prev.iterrows():
        lines.append(f"  [{p['sent_idx']}] {p['text']}")
    lines.append(f">>[{sidx}] {row['text']}")
    for _, n in nxt.iterrows():
        lines.append(f"  [{n['sent_idx']}] {n['text']}")
    lines.append("")

out = "notes/stage4b_mvonly_sheet.txt"
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print(f"wrote {out}: {len(sample)} rows, {len(lines)} lines")
print(sample.groupby("row_kind").size().to_string())
