
from pydantic import BaseModel, Field

from analysis.skill_extractor import SkillMention


class SkillGap(BaseModel):
    skill_name: str

    status: str  # "matched" or "not_found_in_cv"

    importance: str  # "required" or "preferred"

    expected_level: str | None = None

    priority: str  # "high" or "medium"

    candidate_evidence: list[str] = Field(
        default_factory=list
    )

    job_evidence: list[str] = Field(
        default_factory=list
    )


class GapAnalysis(BaseModel):
    total_required_skills: int

    total_preferred_skills: int

    matched_skills: list[SkillGap] = Field(
        default_factory=list
    )

    missing_skills: list[SkillGap] = Field(
        default_factory=list
    )


class GapAnalyzer:

    @staticmethod
    def analyze(
        candidate_skills: list[SkillMention],
        job_skills: list[SkillMention]
    ) -> GapAnalysis:

        # Build a lookup for candidate skills
        candidate_skill_lookup = {
            skill.name.strip().lower(): skill
            for skill in candidate_skills
            if skill.source == "candidate"
        }

        matched_skills = []
        missing_skills = []

        total_required = 0
        total_preferred = 0

        for job_skill in job_skills:

            if job_skill.source != "job":
                continue

            skill_name = job_skill.name

            importance = job_skill.importance or "required"

            if importance == "required":
                total_required += 1
                priority = "high"
            else:
                total_preferred += 1
                priority = "medium"

            # Check whether the candidate has CV evidence
            candidate_skill = candidate_skill_lookup.get(
                skill_name.strip().lower()
            )

            if candidate_skill:

                result = SkillGap(
                    skill_name=skill_name,
                    status="matched",
                    importance=importance,
                    expected_level=job_skill.expected_level,
                    priority=priority,
                    candidate_evidence=candidate_skill.evidence,
                    job_evidence=job_skill.evidence
                )

                matched_skills.append(result)

            else:

                result = SkillGap(
                    skill_name=skill_name,
                    status="not_found_in_cv",
                    importance=importance,
                    expected_level=job_skill.expected_level,
                    priority=priority,
                    candidate_evidence=[],
                    job_evidence=job_skill.evidence
                )

                missing_skills.append(result)

        return GapAnalysis(
            total_required_skills=total_required,
            total_preferred_skills=total_preferred,
            matched_skills=matched_skills,
            missing_skills=missing_skills
        )
