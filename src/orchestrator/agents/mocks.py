"""
Mock agents. Same input/output shape as the real ones, so the whole graph
runs end-to-end before everybody finishes their part.
"""

from __future__ import annotations

import hashlib
from typing import Any

_LEVELS = ["easy", "medium", "hard", "expert"]


def _skill_question_counts(history: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in history:
        question = item["question"]
        # Follow-up questions should not count toward the interview plan.
        if question.get("question_type", "main") != "main":
            continue
        skill = item["question"]["skill"]
        counts[skill] = counts.get(skill, 0) + 1
    return counts


class MockAnalysis:
    def analyze(self, cv_text: str, job_description_text: str) -> dict[str, Any]:
        skills = [
            ("Python", "high", 3, "medium", "matched"),
            ("Machine Learning", "high", 3, "medium", "matched"),
            ("Docker", "medium", 2, "easy", "not_found_in_cv"),
        ]
        planned = [
            {
                "skill_name": name,
                "priority": prio,
                "questions_count": n,
                "initial_difficulty": diff,
                "difficulty_range": ["easy", "medium", "hard"],
                "reason": "mock",
            }
            for name, prio, n, diff, _ in skills
        ]
        gap_entry = lambda s: {  # noqa: E731
            "skill_name": s[0],
            "status": s[4],
            "importance": "required",
            "expected_level": None,
            "priority": s[1],
            "candidate_evidence": [],
            "job_evidence": [],
        }
        return {
            "candidate_profile": {"name": "Mock Candidate"},
            "job_description": {"job_title": "Machine Learning Engineer"},
            "candidate_skills": [],
            "job_skills": [],
            "gap_analysis": {
                "total_required_skills": 3,
                "total_preferred_skills": 0,
                "matched_skills": [gap_entry(s) for s in skills if s[4] == "matched"],
                "missing_skills": [gap_entry(s) for s in skills if s[4] != "matched"],
            },
            "interview_plan": {
                "job_title": "Machine Learning Engineer",
                "total_questions": sum(s[2] for s in skills),
                "planned_skills": planned,
            },
        }

    

def _adjust_difficulty(level, score, difficulty_range):
        """Adjust difficulty based on the candidate's previous score."""
        if not difficulty_range:
            return level

        idx = (
            difficulty_range.index(level)
            if level in difficulty_range
            else 0
        )

        if score >= 7:
            idx += 1
        elif score < 4:
            idx -= 1

        # Keep the difficulty within the allowed range.
        idx = max(0, min(len(difficulty_range) - 1, idx))

        return difficulty_range[idx]

def select_next(plan, history, last_score):
    """Which skill / difficulty is next? (walks the plan, adapts to last_score)
    -> (skill_name, difficulty, number_of_this_question_within_skill) or None."""
    if True:
        counts = _skill_question_counts(history)
        for ps in plan["planned_skills"]:
            asked = counts.get(ps["skill_name"], 0)
            if asked >= ps["questions_count"]:
                continue

            rng = ps.get("difficulty_range") or _LEVELS[:3]
            level = ps.get("initial_difficulty", "easy")

            skill_history = [
                item for item in history
                if item["question"]["skill"] == ps["skill_name"]
                and item["question"].get("question_type", "main") == "main"
            ]

            for item in skill_history:
                score = item.get("evaluation", {}).get("score")

                if score is not None:
                    level = _adjust_difficulty(level, score, rng)

            if (
                skill_history
                and skill_history[-1].get("evaluation", {}).get("score") is None
                and history[-1] is skill_history[-1]
                and last_score is not None
            ):
                level = _adjust_difficulty(level, last_score, rng)

            return ps["skill_name"], level, asked + 1

        return None


class MockInterview:

    def next_question(self, plan, history, last_score):
        nxt = select_next(plan, history, last_score)
        if nxt is None:
            return None
        skill, level, n = nxt
        return {
            "id": f"{skill}-{n}".lower().replace(" ", "_"),
            "skill": skill,
            "difficulty": level,
            "text": f"[mock/{level}] Question #{n} about {skill}?",
            "question_type": "main",
            "parent_question_id": None,
        }

    def follow_up_question(self, question, answer, evaluation, history):
        score = evaluation.get("score", 0)
        missing_points = evaluation.get("missing_points", [])

        # No follow-up is needed if there are no missing points.
        if not missing_points:
            return None

        # Allow only one follow-up per main question.
        for item in history:
            previous_question = item["question"]

            if previous_question.get("parent_question_id") == question["id"]:
                return None

        missing_point = missing_points[0]
        skill = question["skill"]

        if score <= 3:
            difficulty = "easy"
            text = f"Can you explain the basic idea of {missing_point}?"
        elif score <= 6:
            difficulty = "medium"
            text = f"How would you apply {missing_point} in {skill}?"
        else:
            difficulty = "hard"
            text = f"What are the challenges or trade-offs related to {missing_point}?"

        return {
            "id": f"{question['id']}_follow_up",
            "skill": skill,
            "difficulty": difficulty,
            "text": f"[mock/{difficulty}] {text}",
            "question_type": "follow_up",
            "parent_question_id": question["id"],
        }
class MockEvaluation:
    def evaluate(self, question, answer):
        answer = (answer or "").strip()
        if not answer:
            return {"score": 0.0, "feedback": "Empty answer.",
                    "correct_points": [], "missing_points": ["Any answer"]}
        # deterministic pseudo-score from answer length, 3..9
        score = min(9.0, 3.0 + len(answer) / 40)
        return {
            "score": round(score, 1),
            "feedback": f"Mock feedback for {question['skill']}.",
            "correct_points": ["mock correct point"],
            "missing_points": ["mock missing point"] if score < 8 else [],
        }


class MockLearning:
    def recommend(self, payload):
        weak = [s for s, v in payload["skill_scores"].items() if v.get("weak")]
        topics = weak or ["Advanced topics in your target field"]
        return {
            "text": "Mock learning plan: " + ", ".join(topics),
            "recommended_topics": topics,
        }


class MockReport:
    def generate(self, payload):
        scores = payload["skill_scores"]
        strengths = [s for s, v in scores.items() if not v.get("weak")]
        weaknesses = [s for s, v in scores.items() if v.get("weak")]
        md = (
            f"# Interview Report - {payload.get('job_title')}\n\n"
            f"**Strengths:** {', '.join(strengths) or '-'}\n\n"
            f"**Weaknesses:** {', '.join(weaknesses) or '-'}\n\n"
        )
        return {"strengths": strengths, "weaknesses": weaknesses, "markdown": md}
