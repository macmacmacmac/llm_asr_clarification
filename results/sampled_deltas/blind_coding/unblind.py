"""
Unblind the four coders' results with key.json and tabulate per teacher metric: how many top-20 lines
were coded mistranscribed, important (Yes / Partly), and both. Also reports agreement between coders
on (line, question) pairs that appeared in more than one packet.

Run from the repo root after coding_A..D.json exist:
    python results/sampled_deltas/blind_coding/unblind.py MEETING
Writes results/sampled_deltas/blind_coding/MEETING/results.md (and prints it).
"""

import sys
import json
from collections import defaultdict

import pandas as pd

M = sys.argv[1] if len(sys.argv) > 1 else "ES2005d"
D = f"results/sampled_deltas/blind_coding/{M}"
ORDER = ["Sampled 8B: expected total score", "Sampled 8B: max over questions",
         "Logp 8B: max over questions", "Logp 32B: max over questions"]

with open(f"{D}/key.json") as f:
    key = json.load(f)

rows = []
for letter, metric in key["letters"].items():
    with open(f"{D}/coding_{letter}.json") as f:
        coded = json.load(f)
    for c in coded:
        k = key["rows"][c["row"]]
        rows.append({"metric": metric, "coder": letter, "row": c["row"], "rank": k["rank"], "line": k["line"],
                     "q": k["q"], "mistranscribed": c["mistranscribed"], "important": c["important"], "note": c["note"]})
df = pd.DataFrame(rows)

summary = []
for metric in ORDER:
    s = df[df.metric == metric]
    mis = s.mistranscribed == "Yes"
    imp_yes = s.important == "Yes"
    imp_any = s.important.isin(["Yes", "Partly"])
    summary.append({
        "metric": metric,
        "mistranscribed": int(mis.sum()),
        "important: Yes": int(imp_yes.sum()),
        "important: Partly": int((s.important == "Partly").sum()),
        "mistranscribed AND important Yes": int((mis & imp_yes).sum()),
        "mistranscribed AND important Yes/Partly": int((mis & imp_any).sum()),
        "in top 10: mistranscribed AND important Yes": int((mis & imp_yes & (s["rank"] <= 10)).sum()),
    })

pairs = defaultdict(list)
for r in rows:
    pairs[(r["line"], r["q"])].append(r)
shared = {k: v for k, v in pairs.items() if len(v) > 1}
agree_rows = []
for (line, q), v in sorted(shared.items()):
    agree_rows.append({
        "line": line, "q": q, "coders": "".join(sorted(r["coder"] for r in v)),
        "mistranscribed": " / ".join(r["mistranscribed"] for r in v),
        "important": " / ".join(r["important"] for r in v),
        "mis agree": len({r["mistranscribed"] for r in v}) == 1,
        "imp agree": len({r["important"] for r in v}) == 1,
    })
agree = pd.DataFrame(agree_rows, columns=["line", "q", "coders", "mistranscribed", "important", "mis agree", "imp agree"])

md = [f"# Blind coding results: {M} top-20 lines per teacher metric", "",
      "Each metric's top 20 was coded by a separate fresh agent that saw only the quiz and shuffled "
      "(question, ASR text, GT text) rows, with no metric names, ranks, line numbers or scores. "
      "Counts are out of 20 lines per metric.", "",
      pd.DataFrame(summary).to_markdown(index=False), "",
      f"## Agreement on (line, question) pairs coded by more than one coder ({len(agree)} pairs)", "",
      f"mistranscribed agreement: {agree['mis agree'].sum()}/{len(agree)}; "
      f"important agreement: {agree['imp agree'].sum()}/{len(agree)}", "",
      agree.to_markdown(index=False), "",
      "## All codings (unblinded)", "",
      df.sort_values(["metric", "rank"]).to_markdown(index=False), ""]
with open(f"{D}/results.md", "w") as f:
    f.write("\n".join(md))
print("\n".join(md[:8 + len(agree) + 6]))
