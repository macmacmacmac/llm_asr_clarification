import os
import argparse
from llm_asr_clarification import get_logger, OpenAIWrapper
from llm_asr_clarification.constants.quiz_prompts import QUIZ_ANSWER_GENERATOR_SYSTEM_PROMPT, QUIZ_ANSWER_GENERATOR_USER_PROMPT
from tqdm.auto import tqdm
import ipdb
import json
from llm_asr_clarification.constants import SAMPLE_MEETINGS


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
        transcript_path = os.path.join(meeting_path, "transcripts", f"{args.transcript_file}.txt")
        question_path = os.path.join(meeting_path, "transcripts", f"quiz_from_{args.question_file}.json")
        
        logger.info(f"processing: {transcript_path}")
        
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

    
        system_prompt = QUIZ_ANSWER_GENERATOR_SYSTEM_PROMPT.format(
            num_questions=num_questions
        )
        user_prompt = QUIZ_ANSWER_GENERATOR_USER_PROMPT.format(
            transcript=transcript_text,
            questions=formatted_questions,
            num_questions=num_questions
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

            # Set new answers
            for qc, a in zip(question_answers, answers):
                qc[f"answer_using_{args.transcript_file}"] = a
            
            with open(question_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(question_answers, indent=4))

            logger.info("success! Generated answers")

        except AssertionError as e:
            logger.info("Encountered an error")
            logger.error(str(e))

    