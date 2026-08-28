from .get_logger import get_logger
# TODO: Revisit these shortcut imports. They force eager evaluation of heavy ML dependencies
# (like pandas, torch) which breaks minimal environments (like .vllm_venv) that only need the logger.
# Using try/except as a temporary workaround.
try:
    from .models.OpenAIWrapper import OpenAIWrapper
except ImportError:
    pass