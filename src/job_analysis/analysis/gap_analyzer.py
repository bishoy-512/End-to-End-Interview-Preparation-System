from pydantic import BaseModel, Field

from analysis.skill_extractor import SkillMention
from analysis.skill_normalizer import related_skills

IMPORTANCE_WEIGHT = {"required": 3.0, "preferred": 1.0}
STATUS_CREDIT = {"matched": 1.0, "partial": 0.5, "not_found_in_cv": 0.0}


class SkillGap(BaseModel):
    skill_name: str

    # "matched", "partial" (only related skills on the CV) or "not_found_in_cv"
    status: str

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

    # New: skills the CV only covers through related skills.
    partial_skills: list[SkillGap] = Field(
        default_factory=list
    )

    # New: weighted match percentage (required x3, preferred x1, partial 0.5).
    match_score: float = 0.0


class GapAnalyzer:

    @staticmethod
    def analyze(
        candidate_skills: list[SkillMention],
        job_skills: list[SkillMention]
    ) -> GapAnalysis:

        candidate_skill_lookup = {
            skill.name.strip().lower(): skill
            for skill in candidate_skills
            if skill.source == "candidate"
        }

        matched_skills = []
        partial_skills = []
        missing_skills = []

        total_required = 0
        total_preferred = 0
        weight_total = 0.0
        weight_earned = 0.0

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

            candidate_skill = candidate_skill_lookup.get(
                skill_name.strip().lower()
            )

            if candidate_skill:
                status = "matched"
                candidate_evidence = list(candidate_skill.evidence)
            else:
                related = [
                    name for name in related_skills(skill_name)
                    if name.lower() in candidate_skill_lookup
                ]
                if related:
                    status = "partial"
                    candidate_evidence = [
                        f"Related skill on CV: {name}" for name in related
                    ]
                else:
                    status = "not_found_in_cv"
                    candidate_evidence = []

            gap = SkillGap(
                skill_name=skill_name,
                status=status,
                importance=importance,
                expected_level=job_skill.expected_level,
                priority=priority,
                candidate_evidence=candidate_evidence,
                job_evidence=job_skill.evidence
            )

            if status == "matched":
                matched_skills.append(gap)
            elif status == "partial":
                partial_skills.append(gap)
            else:
                missing_skills.append(gap)

            weight = IMPORTANCE_WEIGHT.get(importance, 1.0)
            weight_total += weight
            weight_earned += weight * STATUS_CREDIT[status]

        match_score = (
            round(100 * weight_earned / weight_total, 1)
            if weight_total else 0.0
        )

        return GapAnalysis(
            total_required_skills=total_required,
            total_preferred_skills=total_preferred,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            partial_skills=partial_skills,
            match_score=match_score
        )
