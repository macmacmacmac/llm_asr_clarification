from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  

# -----------------------------
# LLM call
# -----------------------------
class OpenAIWrapper:
    """
        A wrapper around OpenAI's API for convenience. 
        
        Expects `OPENAI_API_KEY` inside the project root.
        
        Usage example
        --------
        >>> from llm_asr_clarification import OpenAIWrapper

        >>> chatgpt = OpenAIWrapper()
        >>> plain_str_answer = chatgpt.prompt_chatgpt(args.prompt)
    """
    def __init__(self, system_prompt: str = "You are a helpful assistant. Follow the task exactly"):
        self.system_prompt = system_prompt
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please add `OPENAI_API_KEY` inside a .env file in project root")

        self.client = OpenAI(api_key=api_key)
    
    def prompt_chatgpt(
        self,
        prompt: str,
        model_name = 'gpt-4o-mini',
        **kwargs
    ):
        defaults = {
            'model': model_name,
            'temperature': 0.0,
            'max_tokens': 128
        }
        params = {**defaults, **kwargs}
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            **params
        )

        return response.choices[0].message.content.strip()
