
from pathlib import Path

from schemas.job import JobDescription
from services.groq_client import generate_structured_output


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "jd_extraction.txt"
)


def load_jd_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


class JDParser:

    @staticmethod
    def parse_text(jd_text: str) -> JobDescription:
        if not jd_text or not jd_text.strip():
            raise ValueError("Job Description text is empty.")

        prompt = load_jd_prompt()

        return generate_structured_output(
            system_prompt=prompt,
            user_prompt=(
                "Extract the job information from this Job Description:\n\n"
                f"{jd_text}"
            ),
            response_schema=JobDescription
        )
