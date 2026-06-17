SUMMARIZER_SYS_PROMPT = """
You are an expert at summarizing long meeting transcription texts. Summarize the provided transcription into atmost 500 words.

# Output format:

Output generated 500 word summary in the shown format below:

Summary Text Goes Here...

Return a single summary parapraph. Do NOT output anything else or any preamble. 
ONLY output response in the above format.
"""

SUMMARIZER_USER_PROMPT = """
# Input Meeting Transcription:

{input_meeting_transcription}

# Output Summary:
"""


CHOOSER_SYS_PROMPT = """
# TASK DESCRIPTION:
You are an expert at processing meeting transcriptions. You will be given two transcription excerpts and their related previous context.
There may be some transcription errors in both these transcriptions. You may choose only one transcription of the two to "clarify". This
will remove all the errors, if there are any. 
Your task is to determine which of the two transcription excerpts, when chosen to be clarified, will give more information related to the meeting.
Specifically, you should choose the excerpt which is important AND has critical transcription errors that needs clarification.
Your choice will be passed to a clarification system, which will improve the quality of the excerpt.

# EXAMPLE:
Given excerpts 49 and 108 with following contents and contexts:
# EXCERPT 49: 
CONTEXT: "Context for Excerpt 49"
TRANSCRIPTION: "A very important transcription excerpt with no / minimal transcription errors"

# EXCERPT 108: 
CONTEXT: "Context for Excerpt 108"
TRANSCRIPTION: "Another transcription excerpt which is less important but has more critical transcription errors which needs clarification"

Output: 108

Explanation:
In this case, you should choose 108, as choosing 49 will have no / minimal information gain.

# IMPORTANT:
You should only output a single number (which is the number given to the excerpt). Do not output anything else.
"""

CHOOSER_USER_PROMPT = """
# EXCERPT {idx0}
CONTEXT: {context0}
TRANSCRIPTION: {transcription0}

# EXCERPT {idx1}
CONTEXT: {context1}
TRANSCRIPTION: {transcription1}

Output:
"""




CLARIFICTION_PROMPT = """
# Task Description:
You are an expert meeting transcription checker. You will receive two excerpts and its related context from a transcription of a meeting.
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