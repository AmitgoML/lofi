import os
from functools import lru_cache
from openai import OpenAI

@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    return OpenAI(api_key=api_key)
