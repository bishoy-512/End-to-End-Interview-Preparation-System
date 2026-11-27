import json
import logging
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

MAX_RETRIES = 2  # extra attempts after the first one


@lru_cache(maxsize=1)
def get_client():
    """
    Create the Groq client on first use, not at import time,
    so the rest of the pipeline can be imported and tested without a key.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to the .env file."
        )

    from groq import Groq

    return Groq(api_key=api_key)


def _clean_json(content: str) -> str:
    """Remove <think> blocks and Markdown code fences if the model adds them."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(
        r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE
    )
    return content.strip()


def generate_structured_output(
    system_prompt: str,
    user_prompt: str,
    response_schema: type[BaseModel]
) -> BaseModel:
    """
    Generate structured output using Groq
    and validate it using a Pydantic schema.
    Retries when the answer is empty, not JSON, or fails validation.
    """

    schema = response_schema.model_json_schema()

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n"
                "Return valid JSON matching this schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            )
        }
    ]

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )

        content = response.choices[0].message.content

        try:
            if not content:
                raise ValueError("Groq returned an empty response.")

            data = json.loads(_clean_json(content))
            return response_schema.model_validate(data)

        except (ValueError, ValidationError) as error:
            last_error = error
            logger.warning(
                "Invalid model output (attempt %d): %s", attempt, error
            )

    raise ValueError(
        f"Failed to parse or validate Groq response: {last_error}"
    ) from last_error
