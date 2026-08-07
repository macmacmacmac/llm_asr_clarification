import os
import argparse
from llm_asr_clarification import get_logger, OpenAIWrapper
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
import re
import ast
import ipdb
import json
from filelock import FileLock


QUIZ_SCORER_SYSTEM_PROMPT = """You are an expert quiz grader. 
Your task is to score a single predicted answer against a correct answer by evaluating if it conveys the same core meaning, ignoring exact wording.

Scoring Rules:
- Award 1 (Correct): The predicted answer paraphrases, captures the essential meaning, or contains the core idea (even with extraneous info or different granularity). Focus on meaning.
- Award 0 (Incorrect): The predicted answer states a fundamentally different fact, contradicts the correct answer, is too vague, or says "I don't know".

Output Format:
Return ONLY a single JSON object. Do not include any preamble or extra text. Use the exact format below:
{
  "score": 0 | 1
}"""


QUIZ_SCORER_USER_PROMPT_TEMPLATE = """Question: {question}
Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}

Output JSON Score:
"""



# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing=================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_use", type=str, default="gpt-4o-mini")
    parser.add_argument("--ami_path", type=str, default="./shared/datasets/amicorpus/train")
    parser.add_argument("--transcript_file", type=str, default="whisper_tiny_diarized_transcript")
    parser.add_argument("--question_file", type=str, default="parsed_diarized_gt")
    parser.add_argument("--do_all_meetings", action="store_true")
    parser.add_argument("--meeting_name", type=str, default="ES2005d")
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

    # Global Variables
    chatgpt = OpenAIWrapper(logger=logger, system_prompt=QUIZ_SCORER_SYSTEM_PROMPT)

    #==============================================================================================

    # determine directories of meetings
    if args.do_all_meetings:
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path)]
    else:
        meeting_paths = [f"{args.ami_path}/{args.meeting_name}"] 

    # For each meeting
    for meeting_path in tqdm(meeting_paths):

        # Fetch the quiz
        quiz_path = os.path.join(meeting_path, "quiz", f"quiz_from_{args.question_file}.json")

        # In case quiz gen failed earlier
        try:
            # Read quiz
            with open(quiz_path, "r", encoding="utf-8") as f:
                quiz = f.read()
        except Exception as e:
            logger.error(f"Something failed, couldnt read question file: {e}")
            continue 

        # Load the quiz as a JSON
        quiz = json.loads(quiz)
        # num_questions = len(quiz)

        for q in quiz:
            user_prompt = QUIZ_SCORER_USER_PROMPT_TEMPLATE.format(
                question = q['question'],
                correct_answer = q['correct_answer'],
                predicted_answer = q[f'answer_using_{args.transcript_file}']
            )

            response = chatgpt.prompt_chatgpt(
                prompt = user_prompt,
                max_completion_tokens = 1024,
                model = args.model_to_use
            )

            # Parse the response
            try:
                response_json = json.loads(response)
                score = response_json["score"]
            except Exception:
                logger.error(
                    f"Could not parse response as JSON. Defaulting to 0.\n"
                    f"Response: {response}"
                )
                score = 0
            
            # Add the score to the quiz
            q[f"score_using_{args.transcript_file}"] = score

        

        # ipdb.set_trace()

        # Save the scored quiz
        # Acquire a lock
        lock = FileLock(f"{quiz_path}.lock")
        with lock:
            with open(quiz_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(quiz, indent=4))


        logger.info("Success! Scored all the questions")
