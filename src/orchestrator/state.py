

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

SCORE_MAX = 10.0
WEAK_SKILL_THRESHOLD = 6.0  # avg score below this => skill is "weak"

Status = Literal[
    "created",
    "analyzing",
    "interviewing",
    "learning",
    "reporting",
    "completed",
    "failed",
]


class Question(TypedDict, total=False):
    id: str
    skill: str
    difficulty: str  # easy | medium | hard | expert
    text: str


class Evaluation(TypedDict, total=False):
    score: float | None  # 0..SCORE_MAX, None if evaluation failed
    feedback: str
    correct_points: list[str]
    missing_points: list[str]
    evaluation_failed: bool


class HistoryItem(TypedDict, total=False):
    question: Question
    answer: str
    evaluation: Evaluation


class InterviewState(TypedDict, total=False):
    # ---- inputs -------------------------------------------------------
    session_id: str
    cv_text: str
    job_description_text: str

    analysis: dict[str, Any]  # full JobAnalysisResult as dict
    plan: dict[str, Any]  # analysis["interview_plan"] (InterviewPlan)

    current_question: Question | None
    pending_answer: str | None
    history: Annotated[list[HistoryItem], operator.add]
    last_score: float | None
    interview_done: bool

    # ---- After the interview -----------------------------------------
    skill_scores: dict[str, dict[str, Any]]  # skill -> {avg, count, scores, weak}
    learning: dict[str, Any]  # (Ahmed) learning plan
    report: dict[str, Any]  # (David) final report

    # ---- Bookkeeping -------------------------------------------------
    status: Status
    errors: Annotated[list[dict[str, Any]], operator.add]
