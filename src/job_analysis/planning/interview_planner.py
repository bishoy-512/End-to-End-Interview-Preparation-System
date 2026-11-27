from math import floor

from schemas.interview_plan_schema import (
    InterviewPlan,
    PlannedSkill
)
from analysis.gap_analyzer import GapAnalysis, SkillGap
from analysis.skill_normalizer import category_of

DEFAULT_MAX_QUESTIONS = 12
STRONG_EVIDENCE = 2  # CV statements that make a matched skill "well supported"

_PRIORITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}
_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
# Inside one priority level, probe gaps before skills the CV already shows.
_GAP_FIRST = {"partial": 0, "not_found_in_cv": 1, "matched": 2}


class InterviewPlanner:

    @staticmethod
    def _sort_key(skill: SkillGap):
        return (
            _PRIORITY_RANK.get(skill.priority, 3),
            _GAP_FIRST.get(skill.status, 3),
            skill.skill_name.lower(),
        )

    @staticmethod
    def _difficulty_and_reason(skill: SkillGap) -> tuple[str, str]:
        if skill.status == "matched":
            if (
                skill.importance == "required"
                and len(skill.candidate_evidence) >= STRONG_EVIDENCE
            ):
                return "hard", (
                    "Required skill with strong CV evidence. "
                    "Questions start harder to test real depth."
                )
            return "medium", (
                "The skill appears in the candidate CV. "
                "Interview questions will verify the "
                "candidate's actual understanding."
            )

        if skill.status == "partial":
            return "easy", (
                "The CV shows only related skills. "
                "Interview questions will check how far they transfer."
            )

        return "easy", (
            "The skill was not found in the CV. "
            "Interview questions will assess whether "
            "the candidate has this knowledge."
        )

    @staticmethod
    def create_plan(
        gap_analysis: GapAnalysis,
        job_title: str | None = None,
        total_questions: int = DEFAULT_MAX_QUESTIONS
    ) -> InterviewPlan:

        if total_questions < 1:
            raise ValueError("total_questions must be at least 1.")

        every_skill = (
            gap_analysis.matched_skills
            + gap_analysis.partial_skills
            + gap_analysis.missing_skills
        )

        # Soft skills (communication, teamwork...) are not technical topics.
        technical = [
            skill for skill in every_skill
            if category_of(skill.skill_name) != "soft_skill"
        ]

        if not technical:
            raise ValueError(
                "Cannot create an interview plan: "
                "No required or preferred technical skills were found."
            )

        technical.sort(key=InterviewPlanner._sort_key)

        # The question budget is a hard cap. Extra skills are reported.
        selected = technical[:total_questions]
        not_covered = [s.skill_name for s in technical[total_questions:]]

        allocations = [1] * len(selected)
        remaining = total_questions - len(selected)

        # Priority sets the share. Gaps get a little extra over matched skills.
        weights = [
            _PRIORITY_WEIGHT.get(skill.priority, 1)
            + (0 if skill.status == "matched" else 1)
            for skill in selected
        ]
        total_weight = sum(weights)

        exact_shares = [remaining * w / total_weight for w in weights]
        base = [floor(share) for share in exact_shares]

        for index, count in enumerate(base):
            allocations[index] += count

        unallocated = total_questions - sum(allocations)
        order = sorted(
            range(len(selected)),
            key=lambda i: (exact_shares[i] - base[i], -i),
            reverse=True
        )
        for index in order[:unallocated]:
            allocations[index] += 1

        planned_skills = []

        for skill, question_count in zip(selected, allocations):
            difficulty, reason = InterviewPlanner._difficulty_and_reason(skill)

            planned_skills.append(
                PlannedSkill(
                    skill_name=skill.skill_name,
                    priority=skill.priority,
                    questions_count=question_count,
                    initial_difficulty=difficulty,
                    difficulty_range=["easy", "medium", "hard"],
                    reason=reason
                )
            )

        return InterviewPlan(
            job_title=job_title,
            total_questions=total_questions,
            planned_skills=planned_skills,
            not_covered=not_covered
        )
