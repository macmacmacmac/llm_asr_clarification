"""
Tally the blind coding across meetings: per teacher metric and meeting, how many of the top-20 lines
were coded mistranscribed, important, and both; plus totals over all meetings and coder agreement on
(line, question) pairs that appeared in more than one packet of the same meeting.

Run from the repo root after unblind.py has been run for every meeting:
    python results/sampled_deltas/blind_coding/tally_all.py
Writes results/sampled_deltas/blind_coding/totals.md (and prints it).
"""

import json
from collections import defaultdict

import torch
import pandas as pd

D = "results/sampled_deltas/blind_coding"
MEETINGS = ["ES2005d", "IN1012", "TS3005d"]
COSINE = "Cosine (GT importance detector)"
ORDER = ["Sampled 8B: expected total score", "Sampled 8B: max over questions",
         "Logp 8B: max over questions", "Logp 32B: max over questions", COSINE]
SHORT = {ORDER[0]: "S8B-exp", ORDER[1]: "S8B-max", ORDER[2]: "LP8B", ORDER[3]: "LP32B", COSINE: "COS"}

rows = []
for m in MEETINGS:
    with open(f"{D}/{m}/key.json") as f:
        key = json.load(f)
    for letter, metric in key["letters"].items():
        with open(f"{D}/{m}/coding_{letter}.json") as f:
            for c in json.load(f):
                k = key["rows"][c["row"]]
                rows.append({"meeting": m, "metric": metric, "coder": letter, "rank": k["rank"], "line": k["line"],
                             "q": k["q"], "mis": c["mistranscribed"] == "Yes", "imp": c["important"]})
df = pd.DataFrame(rows)
df["imp_yes"] = df.imp == "Yes"
df["imp_any"] = df.imp.isin(["Yes", "Partly"])
df["both_yes"] = df.mis & df.imp_yes
df["both_any"] = df.mis & df.imp_any
df["both_yes_top10"] = df.both_yes & (df["rank"] <= 10)


def summarize(g):
    return pd.Series({
        "lines": len(g),
        "mistranscribed": int(g.mis.sum()),
        "important Yes": int(g.imp_yes.sum()),
        "important Partly": int((g.imp == "Partly").sum()),
        "mistranscribed & important Yes": int(g.both_yes.sum()),
        "mistranscribed & important Yes/Partly": int(g.both_any.sum()),
        "top 10: mistranscribed & important Yes": int(g.both_yes_top10.sum()),
    })


totals = df.groupby("metric").apply(summarize).reindex(ORDER).reset_index()
per_meeting = df.pivot_table(index="metric", columns="meeting", values="both_yes", aggfunc="sum").reindex(ORDER)
per_meeting["total"] = per_meeting.sum(axis=1)
per_meeting_any = df.pivot_table(index="metric", columns="meeting", values="both_any", aggfunc="sum").reindex(ORDER)
per_meeting_any["total"] = per_meeting_any.sum(axis=1)

def distinct_questions(col):
    """Distinct (meeting, question) pairs covered by lines flagged in `col`, per metric and meeting."""
    hit = df[df[col]].drop_duplicates(["metric", "meeting", "q"])
    t = hit.pivot_table(index="metric", columns="meeting", values="q", aggfunc="count").reindex(ORDER).reindex(columns=MEETINGS)
    t = t.fillna(0).astype(int)
    t["total"] = t.sum(axis=1)
    return t


distinct_yes = distinct_questions("both_yes")

# Distinct mistranscribed & important (Yes) questions that the metric's own baseline got wrong on the ASR transcript:
# Qwen3-8B (sampled base p(correct) < 0.5, the histogram approximation) for the 8B metrics; the official greedy
# Qwen3-32B quiz run for logp 32B and for cosine (which has no model of its own; 32B is the quiz evaluator).
wrong = {"8B": set(), "32B": set()}
for m in MEETINGS:
    with open(f"shared/datasets/amicorpus/train/{m}/quiz/vllm_quiz.json") as f:
        wrong["32B"] |= {(m, f"q{i}") for i, q in enumerate(json.load(f)) if not q["score_using_custom_transcript_gt_segments"]}
    p_base = torch.load(f"shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7/{m}.pt", weights_only=False)["p_correct"][0]
    wrong["8B"] |= {(m, f"q{i}") for i in range(len(p_base)) if p_base[i] < 0.5}
BASELINE = {metric: ("32B" if metric in ("Logp 32B: max over questions", COSINE) else "8B") for metric in ORDER}
hit = df[df.both_yes].drop_duplicates(["metric", "meeting", "q"])
hit = hit[[(m, q) in wrong[BASELINE[metric]] for metric, m, q in zip(hit.metric, hit.meeting, hit.q)]]
distinct_yes_wrong = hit.groupby("metric").size().reindex(ORDER).fillna(0).astype(int)
num_wrong = {k: len(v) for k, v in wrong.items()}
distinct_any = distinct_questions("both_any")

pairs = defaultdict(list)
for r in rows:
    pairs[(r["meeting"], r["line"], r["q"])].append(r)
shared = [v for v in pairs.values() if len(v) > 1]
mis_agree = sum(len({r["mis"] for r in v}) == 1 for v in shared)
imp_agree = sum(len({r["imp"] for r in v}) == 1 for v in shared)

md = ["# Blind coding totals: pilot meetings ES2005d, IN1012, TS3005d", "",
      "Each metric's top 20 per meeting was coded by a separate fresh agent that saw only the quiz and shuffled "
      "(question, ASR text, GT text) rows. 12 coders in total, 60 lines per metric.", "",
      "## Totals over the 3 meetings", "", totals.to_markdown(index=False), "",
      "## Mistranscribed & important (Yes), per meeting", "", per_meeting.to_markdown(), "",
      "## Mistranscribed & important (Yes or Partly), per meeting", "", per_meeting_any.to_markdown(), "",
      "## Distinct questions covered by mistranscribed & important (Yes) lines (out of 10 per meeting)", "",
      distinct_yes.to_markdown(), "",
      "## Of those, distinct questions the metric's own baseline got wrong on the ASR transcript", "",
      f"Baseline: Qwen3-8B (sampled p < 0.5; {num_wrong['8B']} wrong questions) for S8B-*/LP8B; "
      f"official Qwen3-32B quiz ({num_wrong['32B']} wrong questions) for LP32B and COS.", "",
      distinct_yes_wrong.to_markdown(), "",
      "## Distinct questions covered by mistranscribed & important (Yes or Partly) lines (out of 10 per meeting)", "",
      distinct_any.to_markdown(), "",
      f"## Coder agreement on (meeting, line, question) pairs coded more than once: {len(shared)} pairs", "",
      f"mistranscribed: {mis_agree}/{len(shared)}; important: {imp_agree}/{len(shared)}", ""]
compact = [f"{'':9s}{'Mis':>5s}{'Imp':>6s}{'Mis+Imp':>10s}{'Qs':>6s}{'Qs′':>6s}"]
for metric, row in totals.set_index("metric").iterrows():
    compact.append(f"{SHORT[metric]:9s}{row['mistranscribed']:5d}{row['important Yes']:6d}"
                   f"{row['mistranscribed & important Yes']:10d}{int(distinct_yes.loc[metric, 'total']):6d}"
                   f"{int(distinct_yes_wrong.loc[metric]):6d}")
md += ["## Slack table", "", "```", *compact, "```", ""]

with open(f"{D}/totals.md", "w") as f:
    f.write("\n".join(md))
print("\n".join(md))
