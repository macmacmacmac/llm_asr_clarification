import os
import argparse
from llm_asr_clarification import get_logger
from llm_asr_clarification.models import OracleTranscript

import evaluate
import re
import re
import pandas as pd
import torch
import ipdb
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import re
from rouge_score import rouge_scorer



# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing=================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami_path", type=str, default="./datasets/amicorpus/train")
    parser.add_argument("--transcript_file", type=str, default="whisper_tiny_diarized_transcript")
    parser.add_argument("--do_all_meetings", action="store_true")
    parser.set_defaults(do_all_meetings=False)
    parser.add_argument("--meeting_to_do", type=str, default="./datasets/amicorpus/train/ES2005d")
    
    args, _ = parser.parse_known_args(args_list)

    # Build the logger here
    # first arg is
    logger = get_logger(exp_name)    
    logger.info(
        f"{"="*100}\n\t\t\t\tRunning script: {exp_name}\n{"="*100}"
    )

    # log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(
        f"Received the following arguments:\n{received_args_log}"
    )

    #==============================================================================================
    # directories of meetings
    if args.do_all_meetings:
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path) if entry.name not in ['ami_public_manual_1.6.2', 'xinlu_data']]
    else:
        meeting_paths = [args.meeting_to_do]

    device = "cuda"

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"   # or another causal LM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    with logging_redirect_tqdm(loggers=[logger]):
        for meeting_path in tqdm(meeting_paths):
            file_todo_path = os.path.join(meeting_path, "transcripts", f"{args.transcript_file}.txt")
            file_gt_path = os.path.join(meeting_path, "transcripts", f"parsed_diarized_gt.txt")
            output_preds_path = os.path.join(meeting_path, "artifacts", f"log_probs_{args.transcript_file}.csv")
            
            os.makedirs(os.path.join(meeting_path, "artifacts"), exist_ok=True)

            logger.info(f"I am doing this file: {file_todo_path}")


            with open(file_todo_path, 'r') as f:
                transcript = f.read()

            pattern = (
                r"\((\d+)\s*-\s*(\d+)\)"
                r"\[Speaker ([A-Z])\]:\s*"
                r"(.*)"
            )

            def match(line):
                m = re.match(pattern, line)
                if m:
                    line_dict = {
                        "start_time": int(m.group(1)),
                        "end_time": int(m.group(2)),
                        "speaker": m.group(3),
                        "text": m.group(4),
                    }
                return line_dict

            def get_df_from_transcript(transcript):
                rows = []

                for line in transcript.strip().split("\n"):
                    m = re.match(pattern, line)
                    if m:
                        rows.append({
                            "start_time": int(m.group(1)),
                            "end_time": int(m.group(2)),
                            "speaker": m.group(3),
                            "text": m.group(4),
                        })

                df = pd.DataFrame(rows)
                return df

            df = get_df_from_transcript(transcript)
            df['meeting_name'] = meeting_path.split("/")[-1]
            # ipdb.set_trace()


            # ┌───────────────────────────────────────────────┐
            # │                add gt text to df              │
            # └───────────────────────────────────────────────┘

            # Fills in ground truth column
            gt_transcript = OracleTranscript(file_gt_path, logger='a')
            gts = [gt_transcript.get_oracle_transcription(s_time, e_time) for s_time, e_time in zip(df['start_time'], df['end_time'])]

            pattern = (
                    r"\[Speaker ([A-Z])\]:\s*"
                    r"(.*)"
                )

            def match(line):
                m = re.match(pattern, line)
                if m:
                    line_dict = {
                        "speaker": m.group(1),
                        "text": m.group(2),
                    }
                return line_dict['text']

            gts = ["...".join(match(line) for line in gt) if len(gt) > 0 else None for gt in gts ]
            df['gt_text'] = gts
            df = df.dropna()

            # ┌───────────────────────────────────────────────┐
            # │                add rougeL metrics             │
            # └───────────────────────────────────────────────┘

            scorer = rouge_scorer.RougeScorer(
                ["rougeL"],
                use_stemmer=True
            )

            def rouge_l_precision(pred, ref):
                return scorer.score(
                    ref,   # reference first
                    pred   # prediction second
                )["rougeL"]

            df["rougeL"] = [
                rouge_l_precision(pred, ref).fmeasure
                for pred, ref in zip(df["text"], df["gt_text"])
                
            ]

            df["rougeL_prec"] = [
                rouge_l_precision(pred, ref).precision
                for pred, ref in zip(df["text"], df["gt_text"])
            ]

            # ┌───────────────────────────────────────────────┐
            # │                add logprob metrics            │
            # └───────────────────────────────────────────────┘

            @torch.no_grad()
            def get_logprob_stats(text, col_name):
                """
                Returns:
                    avg_logprob: mean token log prob
                    sum_logprob: total sequence log prob
                    min_logprob: lowest token log prob
                    n_tokens: number of predicted tokens
                """

                enc = tokenizer(text, return_tensors="pt")
                input_ids = enc.input_ids.to(model.device)

                outputs = model(input_ids)

                logits = outputs.logits[:, :-1]
                labels = input_ids[:, 1:]

                log_probs = torch.log_softmax(logits, dim=-1)

                token_log_probs = log_probs.gather(
                    -1,
                    labels.unsqueeze(-1)
                ).squeeze(-1)

                return {
                    f"{col_name}_avg_logprob": token_log_probs.mean().item(),
                    f"{col_name}_sum_logprob": token_log_probs.sum().item(),
                    f"{col_name}_min_logprob": token_log_probs.min().item(),
                    f"{col_name}_n_tokens": token_log_probs.numel(),
                }


            stats = [
                get_logprob_stats(text, 'text')
                for text in (df["text"])
            ]

            # stats = [
            #     get_logprob_stats(text, 'gt_text')
            #     for text in tqdm(df["gt_text"])
            # ]


            stats_df = pd.DataFrame(stats)

            df = pd.concat(
                [df.reset_index(drop=True), stats_df],
                axis=1
            )
            df.to_csv(output_preds_path)
