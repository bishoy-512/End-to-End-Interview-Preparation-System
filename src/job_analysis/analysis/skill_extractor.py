
from pydantic import BaseModel, Field

from schemas.candidate import CandidateProfile
from schemas.job import JobDescription


class SkillMention(BaseModel):
    """
    Represents one skill mentioned in either the CV or the Job Description.
    """

    name: str

    source: str  # "candidate" or "job"

    importance: str | None = None

    expected_level: str | None = None

    evidence: list[str] = Field(default_factory=list)


class SkillExtractor:

    @staticmethod
    def extract_candidate_skills(
        candidate: CandidateProfile
    ) -> list[SkillMention]:

        extracted_skills = []

        # 1. Skills explicitly listed in the CV
        for skill in candidate.skills:
            extracted_skills.append(
                SkillMention(
                    name=skill.name,
                    source="candidate",
                    evidence=skill.evidence
                )
            )

        # 2. Technologies mentioned in work experience
        for experience in candidate.experience:
            for technology in experience.technologies:
                extracted_skills.append(
                    SkillMention(
                        name=technology,
                        source="candidate",
                        evidence=[
                            f"Used in work experience: {experience.job_title}"
                        ]
                    )
                )

        # 3. Technologies mentioned in projects
        for project in candidate.projects:
            for technology in project.technologies:
                extracted_skills.append(
                    SkillMention(
                        name=technology,
                        source="candidate",
                        evidence=[
                            f"Used in project: {project.name}"
                        ]
                    )
                )

        return extracted_skills

    @staticmethod
    def extract_job_skills(job: JobDescription) -> list[SkillMention]:

        extracted_skills = []

        for skill in job.required_skills:
            extracted_skills.append(
                SkillMention(
                    name=skill.name,
                    source="job",
                    importance="required",
                    expected_level=skill.expected_level,
                    evidence=skill.evidence
                )
            )

        for skill in job.preferred_skills:
            extracted_skills.append(
                SkillMention(
                    name=skill.name,
                    source="job",
                    importance="preferred",
                    expected_level=skill.expected_level,
                    evidence=skill.evidence
                )
            )

        return extracted_skills