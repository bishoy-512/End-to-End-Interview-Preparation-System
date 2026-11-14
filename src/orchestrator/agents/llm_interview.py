from __future__ import annotations

from typing import Any, Callable

from .mocks import select_next

SYSTEM_PROMPT = (
    "You are a technical interviewer. Write exactly ONE interview question. "
    "Match the requested skill and difficulty (easy = definitions/basics, "
    "medium = practical application, hard = design/trade-offs/edge cases). "
    "Do not repeat or rephrase any previous question. Do not include the answer."
)


def _default_generate(system_prompt: str, user_prompt: str, schema):
    from .real import JOB_ANALYSIS_DIR, _add_to_path  

    _add_to_path(JOB_ANALYSIS_DIR)
    from services.groq_client import generate_structured_output

    return generate_structured_output(system_prompt, user_prompt, schema)


class LLMInterview:
    def __init__(self, generate: Callable[..., Any] | None = None):
        self._generate = generate or _default_generate

    def next_question(self, plan: dict, history: list[dict], last_score: float | None):
        from pydantic import BaseModel

        class GeneratedQuestion(BaseModel):
            text: str

        nxt = select_next(plan, history, last_score)
        if nxt is None:
            return None
        skill, level, n = nxt

        previous = [h["question"]["text"] for h in history if h["question"]["skill"] == skill]
        prompt = (
            f"Job title: {plan.get('job_title')}\n"
            f"Skill: {skill}\nDifficulty: {level}\n"
            f"Previous questions on this skill (do NOT repeat): {previous or 'none'}\n"
            "Return JSON with a single field: text."
        )
        out = self._generate(SYSTEM_PROMPT, prompt, GeneratedQuestion)
        return {
            "id": f"{skill}-{n}".lower().replace(" ", "_"),
            "skill": skill,
            "difficulty": level,
            "text": out.text.strip(),
        }
