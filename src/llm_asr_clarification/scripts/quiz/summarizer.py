import os
import argparse
from llm_asr_clarification import get_logger, OpenAIWrapper
from tqdm.auto import tqdm
import re
import ipdb
import json
from llm_asr_clarification.constants import SAMPLE_MEETINGS


SUMMARIZER_PROMPT = """# Task
Summarize the following meeting transcript. Preserve the most important details.
Output your summary in readable markdown format, with subheadings for each important aspect discussed.
Provide a high level overview at the top
"""

# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing=================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_use", type=str, default="gpt-4o-mini")
    parser.add_argument("--ami_path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--question_file", type=str, default="parsed_diarized_gt")
    parser.add_argument("--do_all_meetings", action="store_true")
    parser.set_defaults(do_all_meetings=False)
    parser.add_argument("--num_questions", type=int, default=10)

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
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path) if entry.name in SAMPLE_MEETINGS]
    
    for meeting_path in tqdm(meeting_paths):
        file_todo_path = os.path.join(meeting_path, "transcripts", f"{args.question_file}.txt")
        output_preds_path = os.path.join(meeting_path, "summary.md")
        
        logger.info(f"Generating Questions for file: {file_todo_path}")
        
        chatgpt = OpenAIWrapper(system_prompt=SUMMARIZER_PROMPT, logger=logger)

        # Read transcript
        with open(file_todo_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        prompt = f"Meeting transcript: {transcript_text}"

        response_text = chatgpt.prompt_chatgpt(
            prompt, 
            # max_tokens=1024,
            max_completion_tokens=1024,
            model=args.model_to_use
        )
        

        try: 
            with open(output_preds_path, "w", encoding="utf-8") as f:
                f.write(response_text)
        except AssertionError as e:
            logger.error(str(e))
            # f.write("\n\n".join(data))

            # data = [f"Question: {q}\nCorrect Answer: {a}" for q,a in zip(questions, answers)]

        # except Exception:
        #     logger.warning(
        #         f"Could not build response. Probably differing number of questions and answers\n"
        #         f"Questions len: {len(questions)}, Answers len: {len(answers)}"
        #     )



    