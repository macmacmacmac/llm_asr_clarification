# TODO: Revisit these shortcut imports. They force eager evaluation of heavy ML dependencies
# (like pandas, torch) which breaks minimal environments (like .vllm_venv).
# Using try/except as a temporary workaround.
try:
    from .OpenAIWrapper import OpenAIWrapper
except ImportError:
    pass

try:
    from .OracleTranscript import OracleTranscript
except ImportError:
    pass

try:
    from .MistranscriptionDetector import MistranscriptionDetector
except ImportError:
    pass