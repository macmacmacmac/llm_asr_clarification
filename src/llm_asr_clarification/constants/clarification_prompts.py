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


# CHOOSER_SYS_PROMPT = """
# # TASK DESCRIPTION:
# You are an expert at identifying important meeting transcriptions errors. You will be given two transcription 
# excerpts and their related previous context. You task is to choose which of the two transcriptions to
# 'clarify.' When a transcription is 'clarified,' all the mistranscription errors, if there are any, is corrected.

# Because you can only choose one out of the two transcription excerpts to clarify, you should choose the one
# which is more important and has more mistranscriptions. For instance, if a transcription has a lot of errors, but
# is talking about something trivial like lunch, it may not be worth clarifying. Conversely, if a transcription is 
# important, but has almost no errors, then you wouldn't gain any additional information from the clarification. 

# You should choose to clarify the transcription that is most important and has the most errors of the two. 

# The two transcription excerpts will be provided in the following format:

# # FIRST EXCERPT ID: <id>
# CONTEXT: <context>
# TRANSCRIPTION: <transcription>

# # FIRST EXCERPT ID: <id>
# CONTEXT: <context>
# TRANSCRIPTION: <transcription>

# Choose which excerpt is more important and has more errors to clarify by replying with its ID. 

# # IMPORTANT:
# You should only output a single number corresponding to the ID. 
# Do not output any preamble, explanation, or anything else.
# ONLY reply with the ID of your chosen excerpt.
# """
CHOOSER_SYS_PROMPT = """
# TASK DESCRIPTION:
You are an AI assistant optimizing live meeting transcriptions. Your goal is to prevent "alert fatigue" by only asking meeting attendees to clarify transcription errors that actually matter.

You will evaluate two transcription excerpts (along with their preceding context). Your task is to choose the ONE excerpt that is most worth interrupting the user to clarify.

# EVALUATION CRITERIA:
To decide which excerpt wins, weigh the following:
1. Information Value: Does the sentence contain actionable or highly specific information? (e.g., action items, deadlines, names, numbers, technical decisions).
2. Error Severity: Is the transcription error confusing, contradictory, or masking key information?
3. Contextual Recoverability: Can a human easily guess what the garbled text was supposed to say based on the context? If yes, it is NOT worth clarifying.

# OUTPUT FORMAT:
You must output your response in the following format:
<reasoning>
Briefly compare the two excerpts based on the criteria above.
</reasoning>
<choice>
[Insert only the winning excerpt number here]
</choice>

# EXAMPLE:
Given excerpts 49 and 108.

# EXCERPT 49:
CONTEXT: "We are finalizing the deployment schedule for next week."
TRANSCRIPTION: "I think we should push the release to chews day."

# EXCERPT 108:
CONTEXT: "The client asked for a specific budget constraint on the AWS migration."
TRANSCRIPTION: "Yeah the maximum budget is capped at [unintelligible] thousand dollars."

Output:
<reasoning>
Excerpt 49 contains a minor phonetic error ("chews day" instead of "Tuesday") that is easily understood by any reader; it does not need clarification. Excerpt 108 is highly important (budget) and contains a critical error masking the exact number, which cannot be deduced from context.
</reasoning>
<choice>
108
</choice>
"""

CHOOSER_USER_PROMPT = """
# FIRST EXCERPT ID: {idx0}
CONTEXT: {context0}
TRANSCRIPTION: {transcription0}

# SECOND EXCERPT ID: {idx1}
CONTEXT: {context1}
TRANSCRIPTION: {transcription1}

Output:
"""


CHOOSER_SYS_PROMPT_NO_CTX = """
# TASK DESCRIPTION:
You are an expert at identifying important meeting transcriptions errors. You will be given two transcription 
excerpts. You task is to choose which of the two transcriptions to
'clarify.' When a transcription is 'clarified,' all the mistranscription errors, if there are any, is corrected.

Because you can only choose one out of the two transcription excerpts to clarify, you should choose the one
which is more important and has more mistranscriptions. For instance, if a transcription has a lot of errors, but
is talking about something trivial like lunch, it may not be worth clarifying. Conversely, if a transcription is 
important, but has almost no errors, then you wouldn't gain any additional information from the clarification. 

You should choose to clarify the transcription that is most important and has the most errors of the two. 

The two transcription excerpts will be provided in the following format:

# FIRST EXCERPT ID: <id>
TRANSCRIPTION: <transcription>

# FIRST EXCERPT ID: <id>
TRANSCRIPTION: <transcription>

Choose which excerpt is more important and has more errors to clarify by replying with its ID. 

# IMPORTANT:
You should only output a single number corresponding to the ID. 
Do not output any preamble, explanation, or anything else.
ONLY reply with the ID of your chosen excerpt.
"""

CHOOSER_USER_PROMPT_NO_CTX = """
# FIRST EXCERPT ID: {idx0}
TRANSCRIPTION: {transcription0}

# SECOND EXCERPT ID: {idx1}
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