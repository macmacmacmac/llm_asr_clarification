AMBIGUITY_PROMPT = """# Task Description:
You are an expert meeting transcription checker. You will receive an excerpt from a transcription of a meeting.
Your task is to identify which parts of the meeting have likely been mistranscribed and is missing some crucial context.
Focus on **material mistranscriptions** as opposed to trivial errors.

## Logical example:

- "Van said he was going to handle that" ---> likely a mistranscription, 
    but with context its obvious who this is talking about. This should therefore NOT be flagged.
- "Van maid bee bus boing to bundle fat" ---> likely a mistranscription, 
    and huge audio is completely missing and may be important. This should therefore be flagged.

## Output format:

Output your response for whether or not there is a material mistranscription 
in the shown excerpt in JSON format like so:

{{
  "has_material_mistranscription": boolean
}}

Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the following format.

# Input:

{transcript_excerpt}

# Output:

"""

QUIZ_QUESTION_GENERATOR_CHUNKED_PROMPT = """# Task Description:
You are an expert at making quiz questions to test if people have been paying attention to meetings.
You will be shown a transcription taken from a meeting. Your task is to generate a quiz question
that only someone who has paid close attention to the meeting will be able to answer.
The question should not be answerable from common sense, it should instead quiz for specific information that
only someone paying attention to the meeting would be able to answer.

## Output Format:

Output your quiz question in JSON format like so, 
include both the question and the correct answer

{{
  "quiz_question": string,
  "correct_answer": string
}}

Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the following format.

# Input Transcript:

{transcript_excerpt}

# Output Quiz Question:

"""

QUIZ_QUESTION_GENERATOR_PROMPT = """# Task Description:
You are an expert at making quiz questions to test if people have been paying attention to meetings.
You will be shown a transcription taken from a meeting. Your task is to generate {num_questions} quiz questions
that only someone who has paid close attention to the meeting will be able to answer.
The {num_questions} questions should not be answerable from common sense, it should instead quiz for 
important, specific information that only someone paying attention to the meeting would be able to answer.

Avoid making reference to a specific speaker identifier IE: "What did Speaker 1 say" or "What did Speaker A say" etc.

## Output Format:

Output your {num_questions} quiz questions and their corresponding answers in JSON format
as two parallel arrays of strings like so. THERE SHOULD BE PRECISELY {num_questions} QUESTIONS AND ANSWERS. 

{{
  "quiz_questions": [string, string, ...],
  "correct_answers": [string, string, ...]
}}

Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the following format.

# Input Transcript:

{transcript}

# Output JSON of Questions and Answers:

"""

QUIZ_ANSWER_GENERATOR_PROMPT = """# Task Description:
You are an expert at paying attention to meetings and answering quizzes meant to test your understanding those meetings.
You will be shown a transcription taken from a meeting. Your task is to answer {num_questions} quiz questions
that only someone who has paid close attention to the meeting will be able to answer.

## Output Format:

Output your {num_questions} quiz answers in JSON format
as an array of strings like so. THERE SHOULD BE PRECISELY {num_questions} ANSWERS. 

{{
  "answers": [string, string, ...]
}}

Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the following format.

# Input Transcript and Questions:

## Transcript:
{transcript}

## Questions:
{questions}

# Output JSON of Answers:

"""

QUIZ_SCORER_PROMPT = """# Task Description:
You are an expert at grading quizzes. You will be shown {num_questions} quiz questions along with the
corresponding correct answers and the predicted answers. Your task is to determine if the predicted answer
contains the same idea as the correct answer while tolerating paraphrasals. 
If the answer is correct, give a score of 1. Else give a score of 0.

## Output Format:

Output your scores in JSON format like so:

{{
  "scores": [int, int, ...]
}}

Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the following format.

# Input Questions, Correct Answers, and Predicted Answers

{quiz}

# Output JSON of Answers:

"""


