
from pydantic import BaseModel, Field


class JobSkill(BaseModel):
    name: str

    expected_level: str | None = None

    evidence: list[str] = Field(default_factory=list)


class JobDescription(BaseModel):
    job_title: str | None = None

    company: str | None = None

    location: str | None = None

    employment_type: str | None = None

    job_summary: str | None = None

    responsibilities: list[str] = Field(default_factory=list)

    required_skills: list[JobSkill] = Field(
        default_factory=list
    )

    preferred_skills: list[JobSkill] = Field(
        default_factory=list
    )

    required_education: list[str] = Field(
        default_factory=list
    )

    required_experience_years: int | None = Field(
        default=None,
        ge=0
    )

    preferred_experience_years: int | None = Field(
        default=None,
        ge=0
    )

    qualifications: list[str] = Field(
        default_factory=list
    )

    technologies: list[str] = Field(
        default_factory=list
    )

    eligibility_requirements: list[str] = Field(
        default_factory=list
    )
