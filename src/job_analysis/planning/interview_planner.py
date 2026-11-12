
from math import floor

from schemas.interview_plan_schema import (
    InterviewPlan,
    PlannedSkill
)
from analysis.gap_analyzer import GapAnalysis


class InterviewPlanner:

    @staticmethod
    def create_plan(
        gap_analysis: GapAnalysis,
        job_title: str | None = None,
        total_questions: int = 15
    ) -> InterviewPlan:

        all_skills = (
            gap_analysis.matched_skills
            + gap_analysis.missing_skills
        )

        if not all_skills:
            raise ValueError(
                "Cannot create an interview plan: "
                "No required or preferred skills were found."
            )

        # Ensure at least one question per selected skill.
        total_questions = max(
            total_questions,
            len(all_skills)
        )

        # Allocate one question to every skill first.
        allocations = [1] * len(all_skills)

        remaining = total_questions - len(all_skills)

        # Higher priority receives a larger share.
        weights = [
            3 if skill.priority == "high" else
            2 if skill.priority == "medium" else 1
            for skill in all_skills
        ]

        total_weight = sum(weights)

        exact_shares = [
            remaining * weight / total_weight
            for weight in weights
        ]

        extra_questions = [
            floor(share)
            for share in exact_shares
        ]

        for index, count in enumerate(extra_questions):
            allocations[index] += count

        # Distribute remaining questions using largest remainder.
        unallocated = total_questions - sum(allocations)

        remainders = [
            exact_shares[i] - extra_questions[i]
            for i in range(len(all_skills))
        ]

        order = sorted(
            range(len(all_skills)),
            key=lambda i: remainders[i],
            reverse=True
        )

        for index in order[:unallocated]:
            allocations[index] += 1

        planned_skills = []

        for skill, question_count in zip(
            all_skills,
            allocations
        ):

            if skill.status == "matched":
                initial_difficulty = "medium"
                reason = (
                    "The skill appears in the candidate CV. "
                    "Interview questions will verify the "
                    "candidate's actual understanding."
                )

            else:
                initial_difficulty = "easy"
                reason = (
                    "The skill was not found in the CV. "
                    "Interview questions will assess whether "
                    "the candidate has this knowledge."
                )

            planned_skills.append(
                PlannedSkill(
                    skill_name=skill.skill_name,
                    priority=skill.priority,
                    questions_count=question_count,
                    initial_difficulty=initial_difficulty,
                    difficulty_range=[
                        "easy",
                        "medium",
                        "hard"
                    ],
                    reason=reason
                )
            )

        return InterviewPlan(
            job_title=job_title,
            total_questions=total_questions,
            planned_skills=planned_skills
        )
