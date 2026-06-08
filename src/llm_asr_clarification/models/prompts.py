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

QUIZ_QUESTION_GENERATOR_PROMPT = """# Task Description:
You are an expert at making quiz questions to test if people have been paying attention to meetings.
You will be shown a transcription taken from a meeting. Your task is to generate a quiz question
that only someone who has paid close attention to the meeting will be able to answer.
The question should not be answerable from common sense, it should instead quiz for specific information that
only someone paying attention to the meeting would be able to answer.

## Output Format:

Output your quiz question in JSON format like so:

{{
  "quiz_question": string
}}

Return a single JSON object ONLY. Do NOT output anything else or any preamble. 
ONLY output response in the following format.

# Input Transcript:

{transcript_excerpt}

# Output Quiz Question:

"""
