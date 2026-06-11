import os
import argparse
from llm_asr_clarification import get_logger, OpenAIWrapper
from llm_asr_clarification.models.prompts import QUIZ_ANSWER_GENERATOR_PROMPT
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
import re
import ast
import ipdb
import json
# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing=================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--msg", type=str, default="example")
    parser.add_argument("--ami_path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--transcript_file", type=str, default="large_transcript")
    parser.add_argument("--question_file", type=str, default="parsed_gt")
    parser.add_argument("--meeting_to_do", type=str, default="./datasets/amicorpus/ES2005d")
    parser.add_argument("--chunk_size", type=int, default=10)

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
    if args.meeting_to_do:
        meeting_paths = [args.meeting_to_do]
    else:
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path) if 'ami_public_manual_1.6.2' not in entry.name]
    for meeting_path in tqdm(meeting_paths):
        transcript_path = os.path.join(meeting_path, "transcripts", f"{args.transcript_file}.txt")
        question_path = os.path.join(meeting_path, "transcripts", f"quiz_from_{args.question_file}.json")
        
        logger.info(f"I am doing this file: {transcript_path}")
        
        chatgpt = OpenAIWrapper()

        # Read transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        
        # Read question
        with open(question_path, "r", encoding="utf-8") as f:
            question_answers = f.read()

        question_answers = json.loads(question_answers)
        questions = [qa['question'] for qa in question_answers]
        correct_answers = [qa['correct_answer'] for qa in question_answers]

        num_questions = len(questions)
        formatted_questions = [f"Question {i}: {q}\n" for i, q in enumerate(questions)]
        formatted_questions = "".join(formatted_questions)

        prompt = QUIZ_ANSWER_GENERATOR_PROMPT.format(
            transcript=transcript_text,
            questions=formatted_questions,
            num_questions=num_questions
        )

        response_text = chatgpt.prompt_chatgpt(prompt, max_tokens=1024)

        try:
            result = json.loads(response_text)
        except Exception:
            logger.warning(
                f"Could not parse response. Defaulting to None.\n"
                f"Response: {response_text}"
            )
            result = {
                "answers": None,
            }

        try: 
            answers = result.get("answers", None)

            # ipdb.set_trace()

            assert answers is not None, "'answers' is None"
            assert (len(answers) == num_questions), "'answers' has wrong length"

            for qc, a in zip(question_answers, answers):
                qc[f"answer_using_{args.transcript_file}"] = a
            
            with open(question_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(question_answers, indent=4))

            logger.info("success! answered all the questions")
        except AssertionError as e:
            logger.info("Encountered an error")
            logger.error(str(e))

    