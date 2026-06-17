import os
import argparse
from llm_asr_clarification import get_logger, OpenAIWrapper
from llm_asr_clarification.models.prompts import QUIZ_SCORER_PROMPT
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
    parser.add_argument("--model_to_use", type=str, default="gpt-4o-mini")
    parser.add_argument("--ami_path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--transcript_file", type=str, default="qwen_transcript")
    parser.add_argument("--question_file", type=str, default="parsed_diarized_gt")
    parser.add_argument("--do_all_meetings", action="store_true")
    parser.set_defaults(do_all_meetings=False)
    parser.add_argument("--meeting_to_do", type=str, default="/group/jrwhitehill/amicorpus/ES2005d")
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
    if args.do_all_meetings:
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path) if entry.name not in ['ami_public_manual_1.6.2', 'xinlu_data']]
    else:
        meeting_paths = [args.meeting_to_do]

    for meeting_path in tqdm(meeting_paths):
        question_path = os.path.join(meeting_path, "transcripts", f"quiz_from_{args.question_file}.json")

        chatgpt = OpenAIWrapper(logger=logger)

        # Read quiz
        with open(question_path, "r", encoding="utf-8") as f:
            quiz = f.read()

        quiz = json.loads(quiz)
        num_questions = len(quiz)
        template = "Question {i}: {q}\nCorrect Answer: {c}\nPredicted Answer: {a}\n\n"
        # ipdb.set_trace()
        formatted_quiz = [
            template.format(i=i, q=qca.get('question'), c=qca.get("correct_answer"), a=qca.get(f"answer_using_{args.transcript_file}"))
            for i, qca in enumerate(quiz)
        ]
        formatted_quiz = "".join(formatted_quiz)

        prompt = QUIZ_SCORER_PROMPT.format(
            quiz=formatted_quiz,
            num_questions=num_questions
        )

        response_text = chatgpt.prompt_chatgpt(
            prompt, 
            # max_tokens=1024,
            max_completion_tokens=1024,
            model=args.model_to_use
        )

        try:
            result = json.loads(response_text)
        except Exception:
            logger.error(
                f"Could not parse response. Defaulting to None.\n"
                f"Response: {response_text}"
            )

        try: 
            scores = []
            for i in range(num_questions):
                score = result.get(f"question_{i}_score", "n/a")
                scores.append(score)

            # ipdb.set_trace()

            for qca, s in zip(quiz, scores):
                qca[f"score_using_{args.transcript_file}"] = s
            
            with open(question_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(quiz, indent=4))


            logger.info("success! answered all the questions")
        except AssertionError as e:
            logger.info("Encountered an error")
            logger.error(str(e))

    