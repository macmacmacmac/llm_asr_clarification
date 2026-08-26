import os
import argparse
from llm_asr_clarification import get_logger, OpenAIWrapper
from llm_asr_clarification.constants.quiz_prompts import QUIZ_ANSWER_GENERATOR_SYSTEM_PROMPT, QUIZ_ANSWER_GENERATOR_USER_PROMPT
from tqdm.auto import tqdm
import ipdb
import json
from filelock import FileLock


BASELINE_SYS_PROMPT = """# Task Description:
You are an expert at answering quizzes meant to test your understanding on the meetings from the AMI Corpus Dataset.
You will be provided with the name of the relevant meeting from the AMI Corpus Dataset. Your task is to answer {num_questions} quiz questions
that only someone who closely understands that meeting will be able to answer.

## Output Format:

Output your {num_questions} quiz answers in JSON format like so.
THERE SHOULD BE PRECISELY {num_questions} ANSWERS. 

{{
  "question_0_answer": (str) "your answer here",
  "question_1_answer": (str) "your answer here",
  ...
  "question_({num_questions}-1)_answer": (str) "your answer here"
}}

Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the given format.
"""

BASELINE_USR_PROMPT = """# Input AMI Corpus Meeting Name and Questions:

## Meeting Name:
{meeting_name}

## Questions:
{questions}

# Output JSON of Answers:

"""


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_use", type=str, default="gpt-4o-mini")
    parser.add_argument("--ami_path", type=str, default="./shared/datasets/amicorpus/train")
    parser.add_argument("--transcript_file", type=str, default="whisper_tiny_diarized_transcript")
    parser.add_argument("--question_file", type=str, default="parsed_diarized_gt")
    parser.add_argument("--do_all_meetings", action="store_true")
    parser.add_argument("--baseline_prompting", action="store_true")
    parser.add_argument("--meeting_name", type=str, default="ES2005d")
    

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

    # determine directories of meetings
    if args.do_all_meetings:
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path)]
    else:
        meeting_paths = [f"{args.ami_path}/{args.meeting_name}"]       


    for meeting_path in tqdm(meeting_paths):
        # Extract Meeting Name
        meeting_name = meeting_path.split("/")[-1]

        # Prep paths for transcript and quiz
        transcript_path = os.path.join(meeting_path, "transcripts", f"{args.transcript_file}.txt")
        question_path = os.path.join(meeting_path, "quiz", f"quiz_from_{args.question_file}.json")
        
        logger.info(f"processing: {transcript_path}")
        
        # Read transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        
        # In case question gen failed earlier
        try:
            # Read question safely with a lock
            lock = FileLock(f"{question_path}.lock")
            with lock:
                with open(question_path, "r", encoding="utf-8") as f:
                    question_answers = f.read()
        except Exception as e:
            logger.error(f"Something failed, couldnt read question file: {e}")
            continue 
        question_answers = json.loads(question_answers)
        questions = [qa['question'] for qa in question_answers]

        num_questions = len(questions)
        formatted_questions = [f"Question {i}: {q}\n" for i, q in enumerate(questions)]
        formatted_questions = "".join(formatted_questions)

        # If baseline prompting is requested
        if args.baseline_prompting:
            system_prompt = BASELINE_SYS_PROMPT.format(num_questions=num_questions)
            user_prompt = BASELINE_USR_PROMPT.format(
                meeting_name = meeting_name,
                questions = formatted_questions
            )

        # Otherwise use ASR transcript based prompting
        else:
            system_prompt = QUIZ_ANSWER_GENERATOR_SYSTEM_PROMPT.format(
                num_questions=num_questions
            )
            user_prompt = QUIZ_ANSWER_GENERATOR_USER_PROMPT.format(
                transcript=transcript_text,
                questions=formatted_questions
            )


        chatgpt = OpenAIWrapper(
            system_prompt=system_prompt,
            logger=logger
        )

        response_text = chatgpt.prompt_chatgpt(
            user_prompt, 
            # max_tokens=1024,
            max_completion_tokens=1024,
            model=args.model_to_use
        )

        # logger.info(
        #     f"user_prompt:\n\n{user_prompt}\n\n"
        #     f"Response: {response_text}"
        # )

        try:
            result = json.loads(response_text)
        except Exception:
            logger.warning(
                f"Could not parse response.\n"
                f"Response: {response_text}"
            )
        try: 
            answers = []
            for i in range(num_questions):
                answer = result.get(f"question_{i}_answer", "n/a")
                answers.append(answer)

            # logger.info(f"raw answers: {answers}")

            lock = FileLock(f"{question_path}.lock")
            with lock:
                # Re-read to get latest state from any concurrent jobs
                with open(question_path, "r", encoding="utf-8") as f:
                    latest_question_answers = json.loads(f.read())
                
                # Set new answers on the latest state
                for qc, a in zip(latest_question_answers, answers):
                    if args.baseline_prompting:
                        qc["answer_using_baseline_prompting"] = a
                    else:                
                        qc[f"answer_using_{args.transcript_file}"] = a

                with open(question_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(latest_question_answers, indent=4))

            logger.info("success! Generated answers")

        except AssertionError as e:
            logger.info("Encountered an error")
            logger.error(str(e))

    