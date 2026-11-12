
import os
import json

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

client = Groq(api_key=api_key)

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


def generate_structured_output(
    system_prompt: str,
    user_prompt: str,
    response_schema: type[BaseModel]
) -> BaseModel:
    """
    Generate structured output using Groq
    and validate it using a Pydantic schema.
    """

    schema = response_schema.model_json_schema()

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
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
        ],

        response_format={"type": "json_object"},

        temperature=0
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response.")

    try:
        data = json.loads(content)

        return response_schema.model_validate(data)

    except Exception as error:
        raise ValueError(
            f"Failed to parse or validate Groq response: {error}"
        ) from error
