
from typing import Literal

from pydantic import BaseModel, Field, model_validator


DifficultyLevel = Literal[
    "easy",
    "medium",
    "hard",
    "expert"
]

PriorityLevel = Literal[
    "high",
    "medium",
    "low"
]


class PlannedSkill(BaseModel):
    skill_name: str

    priority: PriorityLevel

    questions_count: int = Field(
        ge=0,
        description="Number of interview questions allocated to this skill."
    )

    initial_difficulty: DifficultyLevel = "easy"

    difficulty_range: list[DifficultyLevel] = Field(
        default_factory=lambda: ["easy", "medium", "hard"]
    )

    reason: str = Field(
        description="Why this skill should be assessed during the interview."
    )

    @model_validator(mode="after")
    def validate_difficulty_range(self):
        if not self.difficulty_range:
            raise ValueError("Difficulty range cannot be empty.")

        if self.initial_difficulty not in self.difficulty_range:
            raise ValueError(
                "Initial difficulty must be included in difficulty range."
            )

        return self


class InterviewPlan(BaseModel):
    job_title: str | None = None

    total_questions: int = Field(
        gt=0,
        description="Total number of questions in the interview."
    )

    planned_skills: list[PlannedSkill] = Field(
        min_length=1,
        description="Skills selected for interview assessment."
    )

    @model_validator(mode="after")
    def validate_question_count(self):
        allocated_questions = sum(
            skill.questions_count
            for skill in self.planned_skills
        )

        if allocated_questions != self.total_questions:
            raise ValueError(
                f"Total questions ({self.total_questions}) "
                f"must equal allocated questions ({allocated_questions})."
            )

        return self
