from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
import os
import time
import ipdb
import httpx

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
    def __init__(self, logger, system_prompt: str = "You are a helpful assistant. Follow the task exactly"):
        self.system_prompt = system_prompt
        self.logger = logger
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please add `OPENAI_API_KEY` inside a .env file in project root")

        # 1. Define a hook that physically deletes the cookie header right before sending
        def remove_cookie_header(request: httpx.Request):
            if "cookie" in request.headers:
                del request.headers["cookie"]

        # 2. Assign the event hook to the HTTPX client
        http_client = httpx.Client(
            event_hooks={"request": [remove_cookie_header]}
        )

        self.client = OpenAI(
            api_key=api_key, 
            max_retries=5,
            http_client=http_client
        )
    
    def prompt_chatgpt(
        self,
        prompt: str,
        max_retries: int = 5,
        **kwargs,
    ):
        defaults = {
            "model": "gpt-4o-mini",
        }
        params = {**defaults, **kwargs}

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    **params,
                )

                return response.choices[0].message.content.strip()

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise

                wait_time = 2 ** attempt
                self.logger.warning(
                    f"Rate limited. Retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            
            # except Exception as e:
            #     ipdb.set_trace()
            #     raise
    # def prompt_chatgpt(
    #     self,
    #     prompt: str, 
    #     **kwargs
    # ):
    #     defaults = {
    #         'model': 'gpt-4o-mini',
    #     }
    #     params = {**defaults, **kwargs}
    #     response = self.client.chat.completions.create(
    #         messages=[
    #             {"role": "system", "content": self.system_prompt},
    #             {"role": "user", "content": prompt},
    #         ],
    #         temperature=0,
    #         **params
    #     )

    #     return response.choices[0].message.content.strip()
