"""
Validation-set average quiz accuracy (correct / questions, pooled over the 18 meetings) at each
clarification budget K for ALL + LOGP32B vs ALL + GT (cosine) and ALL + ALL, in the Slack layout used
before. ALL + GT and ALL + ALL come from earlier pipelinev4 runs already stored in vllm_quiz.json.
Also lists fixed / broken question counts vs the ASR transcript.

Run from the repo root:
    python results/sampled_deltas/val_accuracy_table.py
Writes results/sampled_deltas/val_accuracy_table.md (and prints it).
"""

import os
import json

AMI = "shared/datasets/amicorpus/validation"
KS = [10, 20, 30, 40, 50]
# Fresh rerun (logp32b_val_comparison.sh, tag _r2) first; the older stored cosine / ALL rows for reference.
ROWS = [("ALL + COSINE", "gt_r2"), ("ALL + LOGP32B", "logp32b_r2"), ("ALL + ALL", "all_r2"),
        ("COSINE (old)", "gt"), ("ALL + ALL (old)", "all")]
ASR, GT = "score_using_custom_transcript_gt_segments", "score_using_parsed_diarized_gt"

quizzes = {}
for m in sorted(os.listdir(AMI)):
    path = f"{AMI}/{m}/quiz/vllm_quiz.json"
    if os.path.exists(path):
        with open(path) as f:
            quizzes[m] = json.load(f)
questions = [q for quiz in quizzes.values() for q in quiz]
n = len(questions)


def field(key, k):
    return f"{ASR}_all_{key}_{k}_clarify4"


acc_lines = [f"{'':15s}" + "".join(f"{k:>7d}" for k in KS)]
fix_lines = [f"{'fixed/broken':15s}" + "".join(f"{k:>7d}" for k in KS)]
for label, key in ROWS:
    accs, fixes = [], []
    for k in KS:
        f = field(key, k)
        if not all(f in q for q in questions):
            accs.append(f"{'n/a':>7s}")
            fixes.append(f"{'n/a':>7s}")
            continue
        accs.append(f"{sum(q[f] for q in questions) / n:7.3f}")
        fixed = sum((not q[ASR]) and q[GT] and q[f] for q in questions)
        broken = sum(q[ASR] and not q[f] for q in questions)
        fixes.append(f"{f'{fixed}/{broken}':>7s}")
    acc_lines.append(f"{label:15s}" + "".join(accs))
    fix_lines.append(f"{label:15s}" + "".join(fixes))
acc_lines.append(f"{'ASR transcript':15s}{sum(q[ASR] for q in questions) / n:7.3f}")
acc_lines.append(f"{'GT transcript':15s}{sum(q[GT] for q in questions) / n:7.3f}")
fixable = sum((not q[ASR]) and q[GT] for q in questions)

md = [f"# Validation: average quiz accuracy ({len(quizzes)} meetings, {n} questions)", "",
      "Qwen3-32B-FP8 answers + grades greedily; mistranscription detector ALL, so lines are ranked by importance only.",
      "ALL + ALL ranks all lines equally (random ties), i.e. a random-K baseline.",
      "The first three rows are one fresh run (logp32b_val_comparison.sh). \"(old)\" rows and the ASR/GT transcript",
      "rows were stored by earlier runs whose model/prompt is unknown, so treat them only as rough references.", "",
      "```", *acc_lines, "```", "",
      f"Fixable questions (ASR wrong, GT right) fixed / questions ASR got right that broke ({fixable} fixable):", "",
      "```", *fix_lines, "```", ""]
with open("results/sampled_deltas/val_accuracy_table.md", "w") as f:
    f.write("\n".join(md))
print("\n".join(md))
