from __future__ import annotations
import re 
from typing import Any, Callable

from .mocks import select_next

SYSTEM_PROMPT = (
    "You are a technical interviewer. Write exactly ONE interview question. "
    "Match the requested skill and difficulty (easy = definitions/basics, "
    "medium = practical application, hard = design/trade-offs/edge cases). "
    "Do not repeat or rephrase any previous question. Do not include the answer."
    "Use the provided Knowledge Base Context when relevant. "
    "Keep the question focused on the requested skill and difficulty. "
    
)

def _normalize_question(text: str) -> str:
    """Normalize question text to make duplicate comparison easier."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_duplicate(new_question: str, history: list[dict]) -> bool:
    """Check whether the generated question already exists in history."""
    normalized_new = _normalize_question(new_question)

    for item in history:
        previous = item["question"]["text"]

        if normalized_new == _normalize_question(previous):
            return True

    return False


def _default_generate(system_prompt: str, user_prompt: str, schema):
    from .real import JOB_ANALYSIS_DIR, _add_to_path  

    _add_to_path(JOB_ANALYSIS_DIR)
    from services.groq_client import generate_structured_output

    return generate_structured_output(system_prompt, user_prompt, schema)


class LLMInterview:
    def __init__(
        self,
        generate: Callable[..., Any] | None = None,
        retrieve_context: Callable[[str], str] | None = None,
    ):
        self._generate = generate or _default_generate
        self._retrieve_context = retrieve_context

    def next_question(self, plan: dict, history: list[dict], last_score: float | None):
        from pydantic import BaseModel

        class GeneratedQuestion(BaseModel):
            text: str

        nxt = select_next(plan, history, last_score)
        if nxt is None:
            return None
        skill, level, n = nxt

        previous = [h["question"]["text"] for h in history if h["question"]["skill"] == skill]
        retrieved_context = ""

        if self._retrieve_context:
            retrieval_query = (
                f"Job title: {plan.get('job_title')}\n"
                f"Skill: {skill}\n"
                f"Difficulty: {level}\n"
                "Technical interview questions and concepts."
            )

            try:
                retrieved_context = self._retrieve_context(retrieval_query) or ""
            except Exception:
                retrieved_context = ""
        prompt = (
            f"Job title: {plan.get('job_title')}\n"
            f"Skill: {skill}\nDifficulty: {level}\n"
            f"Previous questions on this skill (do NOT repeat): {previous or 'none'}\n"
            f"Knowledge Base Context:\n{retrieved_context or 'No context available.'}\n"
        
            "Return JSON with a single field: text."
        )
        for _ in range(3):
            out = self._generate(SYSTEM_PROMPT, prompt, GeneratedQuestion)
            question_text = out.text.strip()

            if not _is_duplicate(question_text, history):
                return {
                    "id": f"{skill}-{n}".lower().replace(" ", "_"),
                    "skill": skill,
                    "difficulty": level,
                    "text": question_text,
                    "question_type": "main",
                    "parent_question_id": None,
                }

            prompt += (
                f"\nYour previous attempt was a duplicate: {question_text}. "
                "Generate a completely different question."
            )
    
        raise RuntimeError(
            "Could not generate a unique interview question after 3 attempts."
        )
    def follow_up_question(
        self,
        question: dict[str, Any],
        answer: str,
        evaluation: dict[str, Any],
        history: list[dict],
    ):
        from pydantic import BaseModel

        missing_points = evaluation.get("missing_points", [])

        if not missing_points:
            return None

        class GeneratedQuestion(BaseModel):
            text: str

        prompt = (
            f"Original question: {question['text']}\n"
            f"Candidate answer: {answer}\n"
            f"Missing points: {missing_points}\n"
            "Generate exactly ONE focused follow-up question that explores "
            "one of the missing points. Do not repeat the original question "
            "or any previous question. Return JSON with a single field: text."
        )

        follow_up_system_prompt = (
            "You are a technical interviewer. Generate one concise, "
            "relevant follow-up question based on the candidate's answer "
            "and missing knowledge points. Do not provide the answer."
        )

        for _ in range(3):
            out = self._generate(
                follow_up_system_prompt,
                prompt,
                GeneratedQuestion,
            )
            question_text = out.text.strip()

            if not _is_duplicate(question_text, history):
                return {
                    "id": f"{question['id']}_follow_up",
                    "skill": question["skill"],
                    "difficulty": question["difficulty"],
                    "text": question_text,
                    "question_type": "follow_up",
                    "parent_question_id": question["id"],
                }

            prompt += (
                f"\nYour previous attempt was a duplicate: {question_text}. "
                "Generate a completely different follow-up question."
            )

        return None
