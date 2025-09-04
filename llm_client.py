import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("SAMBA_API_KEY"),
    base_url="https://api.sambanova.ai/v1"
)

MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct"

from typing import Generator

def run_llm_query(prompt: str) -> Generator[str, None, None]:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, concise legal assistant. "
                        "Respond only with natural language answers. "
                        "Do not use function calls, tool syntax, or tags like <|python_start|>."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            top_p=0.1,
            max_tokens=1500,
            stream=True
        )
        for chunk in response:
            if hasattr(chunk.choices[0].delta, "content"):
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"[LLM ERROR]: {e}"
