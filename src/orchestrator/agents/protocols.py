from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol
from dotenv import load_dotenv
load_dotenv()

class AnalysisAgent(Protocol):
    def analyze(self, cv_text: str, job_description_text: str) -> dict[str, Any]:
        """-> dict with at least: interview_plan, gap_analysis,
        candidate_skills, job_skills (the JobAnalysisResult as dict)."""


class InterviewAgent(Protocol):
    def next_question(
        self,
        plan: dict[str, Any],
        history: list[dict[str, Any]],
        last_score: float | None,
    ) -> dict[str, Any] | None:
        """-> {id, skill, difficulty, text}, or None when the interview is over."""


class EvaluationAgent(Protocol):
    def evaluate(self, question: dict[str, Any], answer: str) -> dict[str, Any]:
        """-> {score (0..10), feedback, correct_points, missing_points}"""


class LearningAgent(Protocol):
    def recommend(self, payload: dict[str, Any]) -> dict[str, Any]:
        """payload: {job_title, skill_scores, gaps, history, report}
        -> {text, recommended_topics: [str]}"""


class ReportAgent(Protocol):
    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """payload: {job_title, plan, skill_scores, history, analysis}
        -> {strengths: [str], weaknesses: [str], markdown: str, ...}"""


@dataclass
class AgentSet:
    analysis: AnalysisAgent
    interview: InterviewAgent
    evaluation: EvaluationAgent
    learning: LearningAgent
    report: ReportAgent

def build_agents() -> AgentSet:
    """Pick mock / real / llm implementation per agent from environment variables."""
    from . import llm_interview, mocks, real  # local import: real.py must stay lazy

    def pick(var: str, mock_name: str, real_name: str, llm_name: str | None = None):
        mode = os.getenv(var, "mock").strip().lower()
        if mode == "llm" and llm_name:
            return getattr(llm_interview, llm_name)()
        if mode == "real":
            cls = getattr(real, real_name, None)
            if cls is None:
                raise RuntimeError(
                    f"{var}=real but agents/real.py has no class `{real_name}`."
                )
            return cls()
        return getattr(mocks, mock_name)()

    return AgentSet(
        analysis=pick("ORCH_ANALYSIS", "MockAnalysis", "RealAnalysis"),
        interview=pick("ORCH_INTERVIEW", "MockInterview", "RealInterview", "LLMInterview"),
        evaluation=pick("ORCH_EVALUATION", "MockEvaluation", "RealEvaluation"),
        learning=pick("ORCH_LEARNING", "MockLearning", "RealLearning"),
        report=pick("ORCH_REPORT", "MockReport", "RealReport"),
    )
