"""
    CPU post-processing for generate_sampled_deltas.py.

    For every meeting file in --results-dir it computes, per question q and line j,

        z[j, q] = (p[j, q] - median_j p[:, q]) / max(1.4826 * MAD_j p[:, q], 1 / K)

    where p[j, q] is the fraction of the K samples graded correct with line j clarified. Centering
    on the median across lines (not on the base transcript) removes the offset that comes from the
    base being one noisy point; the MAD is a robust scale that the few truly important lines can't
    inflate. The 1/K floor stops the scale from collapsing to 0 when most lines share one value.

    Labels: for each question, take its top-m lines with z >= --z-thresh; a line is important if
    any question picked it (the per-question cap limits one question's redundant evidence lines
    from dominating). No-op lines (GT == ASR) are never labeled.

    It also writes a markdown report with: base-vs-median offset, split-half stability (first K/2
    vs last K/2 samples), and agreement with the logp deltas (--logp-paths) on the same meetings.
"""

import os
import argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from llm_asr_clarification import get_logger


def robust_z(p: np.ndarray, valid: np.ndarray, num_samples: int) -> np.ndarray:
    """p: [N, Q] per-line p(correct). Median/MAD use valid lines only (no-op lines are copies of
    the base). Returns [N, Q] robust z-scores, 0 for invalid lines."""
    med = np.median(p[valid], axis=0)
    mad = np.median(np.abs(p[valid] - med), axis=0) * 1.4826
    scale = np.maximum(mad, 1.0 / num_samples)
    return np.where(valid[:, None], (p - med) / scale, 0.0)


def capped_labels(z: np.ndarray, valid: np.ndarray, top_m: int, z_thresh: float) -> np.ndarray:
    """Per question, label its top-m valid lines with z >= z_thresh. Returns [N] bool."""
    labels = np.zeros(z.shape[0], dtype=bool)
    for q in range(z.shape[1]):
        zq = np.where(valid, z[:, q], -np.inf)
        top = np.argsort(-zq)[:top_m]
        labels[top[zq[top] >= z_thresh]] = True
    return labels


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def top_overlap(a: np.ndarray, b: np.ndarray, n: int) -> int:
    return len(set(np.argsort(-a)[:n]) & set(np.argsort(-b)[:n]))


def dominance(z: np.ndarray, score: np.ndarray, n: int) -> tuple[int, int]:
    """Among the top-n lines by score, (most common argmax question, how many lines it takes)."""
    top = np.argsort(-score)[:n]
    counts = np.bincount(z[top].argmax(axis=1), minlength=z.shape[1])
    return int(counts.argmax()), int(counts.max())


def line_text(line: str, width: int = 90) -> str:
    text = line.split("]: ", 1)[-1].replace("|", "/")
    return text if len(text) <= width else text[:width - 3] + "..."


def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True, help="Directory of per-meeting .pt files from generate_sampled_deltas.py")
    parser.add_argument("--ami-path", type=str, default="./shared/datasets/amicorpus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--logp-paths", type=str, nargs="*", default=[], help="logp delta .pt files to compare against, e.g. ./shared/datasets/logp_deltas/train_32b.pt")
    parser.add_argument("--top-m", type=int, default=3, help="Max lines labeled per question")
    parser.add_argument("--z-thresh", type=float, default=3.0)
    parser.add_argument("--top-n", type=int, default=20, help="Size of the top-line sets used for overlap / dominance")
    parser.add_argument("--show-lines", type=int, default=10, help="Top lines printed per meeting in the report")
    parser.add_argument("--report-dir", type=str, default="./results/sampled_deltas")

    args, _ = parser.parse_known_args(args_list)

    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    meeting_files = sorted(f for f in os.listdir(args.results_dir) if f.endswith(".pt"))
    if not meeting_files:
        logger.error(f"No meeting files found in {args.results_dir}")
        return

    logp_sets = {os.path.basename(p).removesuffix(".pt"): torch.load(p) for p in args.logp_paths}

    derived_dir = os.path.join(args.results_dir, "derived")
    os.makedirs(derived_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    labels_out, soft_out = {}, {}
    summary_rows, comparison_rows, meeting_sections = [], [], []

    for fname in meeting_files:
        meeting_name = fname.removesuffix(".pt")
        data = torch.load(os.path.join(args.results_dir, fname))
        correct = data["correct"].numpy()               # [V, Q, K]
        is_noop = data["is_noop"].numpy()               # [N]
        valid = ~is_noop
        num_q, K = correct.shape[1], correct.shape[2]

        p = correct.mean(axis=-1)                       # [V, Q]
        z = robust_z(p[1:], valid, K)                   # [N, Q]
        score = np.where(valid, z.max(axis=1), -np.inf)
        labels = capped_labels(z, valid, args.top_m, args.z_thresh)

        labels_out[meeting_name] = labels.tolist()
        soft_out[meeting_name] = {
            "robust_z": torch.from_numpy(z),            # [N, Q]
            "line_score": torch.from_numpy(np.where(valid, z.max(axis=1), np.nan)),  # [N] max over questions, NaN for no-op lines
            "delta_vs_base": torch.from_numpy(p[1:] - p[0]),  # [N, Q] for reference only
        }

        # Split-half stability: first K/2 samples vs last K/2 samples
        half = K // 2
        z_a = robust_z(correct[1:, :, :half].mean(axis=-1), valid, half)
        z_b = robust_z(correct[1:, :, half:].mean(axis=-1), valid, K - half)
        score_a = np.where(valid, z_a.max(axis=1), -np.inf)
        score_b = np.where(valid, z_b.max(axis=1), -np.inf)
        labels_a = capped_labels(z_a, valid, args.top_m, args.z_thresh)
        labels_b = capped_labels(z_b, valid, args.top_m, args.z_thresh)
        union = (labels_a | labels_b).sum()
        dom_q, dom_n = dominance(z, score, args.top_n)

        mean_score = data["mean_score"].numpy()
        base_outside = np.mean([
            not (np.quantile(p[1:][valid, q], 0.05) <= p[0, q] <= np.quantile(p[1:][valid, q], 0.95)) for q in range(num_q)
        ])
        summary_rows.append({
            "meeting": meeting_name,
            "lines": len(is_noop),
            "no-op lines": int(is_noop.sum()),
            "K": K,
            "base score": round(float(mean_score[0]), 2),
            "median version score": round(float(np.median(mean_score[1:][valid])), 2),
            "max version score": round(float(mean_score[1:][valid].max()), 2),
            "q with base outside 5-95%": f"{base_outside:.0%}",
            "labeled lines": int(labels.sum()),
            f"top-{args.top_n} dominant q (lines)": f"q{dom_q} ({dom_n})",
            "split-half spearman": round(spearman(score_a[valid], score_b[valid]), 3),
            f"split-half top-{args.top_n} overlap": top_overlap(score_a, score_b, args.top_n),
            "split-half label jaccard": round(float((labels_a & labels_b).sum() / union), 3) if union else float("nan"),
            "truncated answers": data["stats"]["num_truncated_answers"],
            "judge parse failures": data["stats"]["num_score_parse_failures"],
            "answer time (min)": round(data["stats"]["answer_duration_sec"] / 60, 1),
            "score time (min)": round(data["stats"]["score_duration_sec"] / 60, 1),
        })

        for logp_name, logp_data in logp_sets.items():
            if meeting_name not in logp_data:
                continue
            logps = logp_data[meeting_name]["logps"].numpy()[1:]   # [N, Q]
            if logps.shape != z.shape:
                logger.warning(f"[{meeting_name}] logp shape {logps.shape} != sampled shape {z.shape} in {logp_name}, skipping comparison")
                continue
            logp_z = (logps - logps.mean(axis=0)) / logps.std(axis=0)
            logp_score = np.where(valid, logp_z.max(axis=1), -np.inf)
            lq, ln = dominance(logp_z, logp_score, args.top_n)
            comparison_rows.append({
                "meeting": meeting_name,
                "logp file": logp_name,
                "spearman (line score)": round(spearman(score[valid], logp_score[valid]), 3),
                "spearman (per-q z, pooled)": round(spearman(z[valid].ravel(), logp_z[valid].ravel()), 3),
                f"top-{args.top_n} overlap": top_overlap(score, logp_score, args.top_n),
                "logp dominant q (lines)": f"q{lq} ({ln})",
                "sampled dominant q (lines)": f"q{dom_q} ({dom_n})",
            })

        # Top lines with text, for eyeballing
        transcript_dir = os.path.join(args.ami_path, args.split, meeting_name, "transcripts")
        try:
            with open(os.path.join(transcript_dir, "custom_transcript_gt_segments.txt"), "r", encoding="utf-8") as f:
                asr_lines = [line.strip() for line in f]
            with open(os.path.join(transcript_dir, "parsed_diarized_gt.txt"), "r", encoding="utf-8") as f:
                gt_lines = [line.strip() for line in f]
        except FileNotFoundError:
            asr_lines = gt_lines = None

        section = [f"### {meeting_name}", ""]
        if asr_lines is not None:
            section += ["| line | q | z | p base → p line | label | ASR | GT |", "|---|---|---|---|---|---|---|"]
            for j in np.argsort(-score)[:args.show_lines]:
                q = int(z[j].argmax())
                section.append(
                    f"| {j} | q{q} | {z[j, q]:.1f} | {p[0, q]:.2f} → {p[j + 1, q]:.2f} | {'yes' if labels[j] else ''} | {line_text(asr_lines[j])} | {line_text(gt_lines[j])} |"
                )
            section += ["", "Questions:", ""]
            section += [f"- q{i}: {q['question']} — *{q['correct_answer']}*" for i, q in enumerate(data["quiz"])]
        meeting_sections.append("\n".join(section))
        logger.info(f"[{meeting_name}] {int(labels.sum())} labeled lines; top-{args.top_n} dominant q{dom_q} with {dom_n} lines")

    tag = f"topm{args.top_m}_z{args.z_thresh}"
    labels_path = os.path.join(derived_dir, f"labels_{tag}.pt")
    soft_path = os.path.join(derived_dir, "robust_z.pt")
    torch.save(labels_out, labels_path)
    torch.save(soft_out, soft_path)

    report = [
        "# Sampled-correctness importance: pilot report",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from `{args.results_dir}`.",
        "",
        "## What is reported",
        "",
        "- **p(line, q)**: fraction of the K sampled answers graded correct when only that line is clarified.",
        "- **z**: per-question robust z of p across lines, (p − median) / max(1.4826·MAD, 1/K). Line score = max over questions.",
        f"- **Labels**: per question, its top-{args.top_m} lines with z ≥ {args.z_thresh}; union over questions. No-op lines (GT == ASR) are never labeled.",
        "- **Scores** are quiz scores out of the number of questions, averaged over the K samples.",
        "- **Split-half**: z and labels recomputed from the first K/2 and last K/2 samples separately; agreement shows whether K/2 samples would have been enough.",
        "- **Logp comparison**: logp z as in `generate_logp_deltas.py` (z-score across lines, max over questions). No-op lines excluded.",
        "",
        "## Per-meeting summary",
        "",
        pd.DataFrame(summary_rows).to_markdown(index=False),
        "",
    ]
    if comparison_rows:
        report += ["## Agreement with logp deltas", "", pd.DataFrame(comparison_rows).to_markdown(index=False), ""]
    report += [
        "## Top lines per meeting",
        "",
        "\n\n".join(meeting_sections),
        "",
        "## Artifacts",
        "",
        f"- Raw per-meeting outcomes: `{args.results_dir}/<meeting>.pt` (`correct[V,Q,K]`, `p_correct`, `mean_score`, `answer_table`, ...)",
        f"- Binary labels `{{meeting: [bool]}}`: `{labels_path}`",
        f"- Soft scores (`robust_z`, `line_score`, `delta_vs_base`): `{soft_path}`",
    ]
    report_path = os.path.join(args.report_dir, f"report_{os.path.basename(os.path.normpath(args.results_dir))}_{tag}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    logger.info(f"Saved labels to {labels_path}, soft scores to {soft_path}, report to {report_path}")
