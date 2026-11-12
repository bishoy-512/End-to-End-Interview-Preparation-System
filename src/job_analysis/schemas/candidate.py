
from pydantic import BaseModel, Field

from typing import Literal

SkillCategory = Literal[
    "programming_language",
    "ai_concept",
    "machine_learning",
    "framework",
    "library",
    "tool",
    "platform",
    "database",
    "soft_skill",
    "language"
]

class Education(BaseModel):
    degree: str
    institution: str | None = None
    field_of_study: str | None = None
    graduation_year: int | None = None


class WorkExperience(BaseModel):
    job_title: str
    company: str | None = None
    description: str
    technologies: list[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)


class CandidateSkill(BaseModel):
    name: str

    category: SkillCategory

    evidence: list[str] = Field(
        default_factory=list,
        description="Statements or sections supporting this skill."
    )


class CandidateProfile(BaseModel):
    name: str | None = None

    professional_summary: str | None = None

    education: list[Education] = Field(default_factory=list)

    experience: list[WorkExperience] = Field(default_factory=list)

    projects: list[Project] = Field(default_factory=list)

    skills: list[CandidateSkill] = Field(default_factory=list)
