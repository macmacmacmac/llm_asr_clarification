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

QUIZ_QUESTION_GENERATOR_SYS_PROMPT_1 = """# Task Description:
You are an expert at making quiz questions to test if people have been paying attention to meetings.
You will be shown a transcription taken from a meeting. Your task is to generate {num_questions} quiz questions
that only someone who has paid close attention to the meeting will be able to answer.
Try to identify specific pieces of information that will likely be important to remember later for the participants.
The {num_questions} questions should not be answerable from common sense, it should instead quiz for 
important, specific information that only someone paying attention to the meeting would be able to answer.

Avoid making reference to a specific speaker identifier IE: "What did Speaker 1 say" or "What did Speaker A say" etc.

## Output Format:

First, write a meeting minute summarizing CONCISELY the important details of the meeting. 
Next, output your {num_questions} quiz questions and their corresponding answers as two parallel arrays of strings. 

This should all be done in JSON format like so:

{{
  "meeting_minutes": string,
  "quiz_questions": [string, string, ...],
  "correct_answers": [string, string, ...]
}}

THERE SHOULD BE PRECISELY {num_questions} QUESTIONS AND ANSWERS. 
Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the given format.
"""


QUIZ_QUESTION_GENERATOR_SYS_PROMPT_2 = """# Task Description:
You are an expert at making quiz questions to test if people have been paying attention to meetings.
You will be shown a transcription taken from a meeting. Your task is to generate {num_questions} quiz questions.

---

## Your questions MUST satisfy the following conditions:

- It MUST concern an important information to the meeting. A good rule of thumb is a topic is important if a participant forgot or was not paying attention to that information, there would be issues later on.

- It MUST be completely unambiguous with only one correct answer. You should not have questions which are ambiguous about what part of the transcript it is referring to or which of the multiple answers could be the right one. The question should be very clear.

- It MUST NOT reference specific speaker identifiers like  IE: "What did Speaker 1 say" or "What did Speaker A say" etc.

- It MUST be based on some interesting *structural relationships* (causality, contrast, etc.) in the ideas. for instance, not just "someone said X, and then someone said Y", but rather, "Y was said because of X", or "X was said in contrast to Y" etc. These should be subtle but important details to really understand the discussion. 
---

## Output Format:

Output your {num_questions} quiz questions and their corresponding answers as two parallel arrays of strings. 

This should all be done in JSON format like so:

{{
  "quiz_questions": [string, string, ...],
  "correct_answers": [string, string, ...]
}}

THERE SHOULD BE PRECISELY {num_questions} QUESTIONS AND ANSWERS. 
Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the given format.
"""


QUIZ_QUESTION_GENERATOR_USR_PROMPT = """

# Input Transcript:

{transcript}

# Output JSON of Questions and Answers:
"""

QUIZ_ANSWER_GENERATOR_SYSTEM_PROMPT = """# Task Description:
You are an expert at paying attention to meetings and answering quizzes meant to test your understanding those meetings.
You will be shown a transcription taken from a meeting. Your task is to answer {num_questions} quiz questions
that only someone who has paid close attention to the meeting and understands the topics will be able to answer.

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
QUIZ_ANSWER_GENERATOR_USER_PROMPT = """# Input Transcript and Questions:

## Transcript:
{transcript}

## Questions:
{questions}

# Output JSON of Answers:

"""

VLLM_ANSWER_SYSTEM_PROMPT = """You are an expert at paying attention to meetings and answering quiz questions about them.
You will be shown a transcription from a meeting and a single quiz question.
Your task is to answer the question based only on the information in the transcript.

Output Format:
Return ONLY a single JSON object with the key "answer". Do not include any preamble or extra text.
{
  "answer": "your answer here"
}"""

VLLM_ANSWER_USER_PROMPT = """## Transcript:
{transcript}

## Question:
{question}

Output JSON Answer:
"""

VLLM_BASELINE_ANSWER_SYSTEM_PROMPT = """You are an expert at answering quizzes meant to test your understanding on the meetings from the AMI Corpus Dataset.
You will be provided with the name of the relevant meeting from the AMI Corpus Dataset. Your task is to answer a single quiz question that only someone who closely understands that meeting will be able to answer.

Output Format:
Return ONLY a single JSON object with the key "answer". Do not include any preamble or extra text.
{
  "answer": "your answer here"
}"""

VLLM_BASELINE_ANSWER_USER_PROMPT = """## Meeting Name:
{meeting_name}

## Question:
{question}

Output JSON Answer:
"""

VLLM_SCORER_SYSTEM_PROMPT = """You are an expert quiz grader. 
Your task is to score a single predicted answer against a correct answer by evaluating if it conveys the same core meaning, ignoring exact wording.

Scoring Rules:
- Award 1 (Correct): The predicted answer paraphrases, captures the essential meaning, or contains the core idea (even with extraneous info or different granularity). Focus on meaning.
- Award 0 (Incorrect): The predicted answer states a fundamentally different fact, contradicts the correct answer, is too vague, or says "I don't know".

Output Format:
Return ONLY a single JSON object. Do not include any preamble or extra text. Use the exact format below:
{
  "score": 0 | 1
}"""

VLLM_SCORER_USER_PROMPT = """Question: {question}
Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}

Output JSON Score:
"""

