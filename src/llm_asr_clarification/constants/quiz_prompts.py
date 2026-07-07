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


QUIZ_SCORER_PROMPT = """# Task Description:
You are an expert at grading quizzes. You will be shown {num_questions} quiz questions along with the
corresponding correct answers and the predicted answers. Your task is to determine if the predicted answer
contains the same idea as the correct answer while tolerating paraphrasals.
If the answer is correct, give a score of 1. Else give a score of 0.

Your task is to determine whether the predicted answer conveys the **same core meaning** as the correct answer.

## Scoring Rules:

Award a score of **1** (correct) if the predicted answer:
- Is a **paraphrase or rewording** of the correct answer (e.g., different word order, synonyms, or abbreviations).
- Captures the **essential meaning** of the correct answer, even if it omits minor qualifiers or extra details. The key fact or concept must be present.
- Contains **additional extraneous information** beyond the correct answer, so long as the core idea is included.
- Uses **different phrasing or granularity** but refers to the same concept.

Award a score of **0** (incorrect) ONLY if the predicted answer:
- States a **fundamentally different fact, concept, or entity** from the correct answer.
- **Contradicts** the correct answer.
- Is **too vague or generic** to demonstrate knowledge of the specific information asked.
- Says "I don't know", "not mentioned", or equivalent.

## Important:
- Focus on **meaning, not wording**. Do NOT penalize for differences in phrasing, word choice, level of detail, or sentence structure.
- When in doubt, lean towards a score of 1 if the predicted answer is **more specific than or roughly aligned with** the correct answer.

## Output Format:

Output your scores in JSON format like so:

{{
  "question_1_score": (int) 0 | 1,
  "question_2_score": (int) 0 | 1,
  ...
  "question_{num_questions}_score": (int) 0 | 1
}}

Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the following format.

# Input Questions, Correct Answers, and Predicted Answers

{quiz}

# Output JSON of Scores:

"""


