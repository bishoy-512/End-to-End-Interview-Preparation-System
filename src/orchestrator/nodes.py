

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from langgraph.types import interrupt

from .agents import AgentSet
from .state import SCORE_MAX, WEAK_SKILL_THRESHOLD, InterviewState

log = logging.getLogger("orchestrator")

MAX_RETRIES = 2  # extra attempts after the first one
RETRY_DELAY_S = 0.5
FINISH_ACTION = "finish"  # answer payload {"action": "finish"} ends the interview early


def _call(agent_name: str, fn: Callable[..., Any], *args) -> tuple[Any, dict | None]:
    """Run fn with retries. Returns (result, None) or (None, error_dict)."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args), None
        except Exception as exc:  # noqa: BLE001 - agents are third-party code
            last_exc = exc
            log.warning("%s failed (attempt %d): %s", agent_name, attempt + 1, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S * (attempt + 1))
    return None, {
        "agent": agent_name,
        "error": f"{type(last_exc).__name__}: {last_exc}",
        "attempts": MAX_RETRIES + 1,
    }


# ------------------------------------------------------------ fallbacks ---
def _fallback_question(plan: dict, history: list[dict]) -> dict | None:
    counts: dict[str, int] = {}

    for item in history:
        question = item["question"]

        if question.get("question_type", "main") != "main":
            continue

        skill = question["skill"]
        counts[skill] = counts.get(skill, 0) + 1

    for ps in plan["planned_skills"]:
        asked = counts.get(ps["skill_name"], 0)

        if asked < ps["questions_count"]:
            level = ps.get("initial_difficulty", "easy")

            return {
                "id": f"fallback-{ps['skill_name']}-{asked + 1}".lower().replace(" ", "_"),
                "skill": ps["skill_name"],
                "difficulty": level,
                "text": f"Explain your experience with {ps['skill_name']} "
                        f"and describe a concept you consider important.",
                "question_type": "main",
                "parent_question_id": None,
            }

    return None

def _main_question_count(history: list[dict]) -> int:
    return sum(
        1
        for item in history
        if item["question"].get("question_type", "main") == "main"
    )
def aggregate_skill_scores(history: list[dict]) -> dict[str, dict]:
    per_skill: dict[str, list[float]] = {}

    for item in history:
        question = item["question"]

        if question.get("question_type", "main") != "main":
            continue

        skill = question["skill"]
        per_skill.setdefault(skill, [])

        score = (item.get("evaluation") or {}).get("score")

        if score is not None:
            per_skill[skill].append(float(score))

    out = {}

    for skill, scores in per_skill.items():
        avg = sum(scores) / len(scores) if scores else 0.0

        out[skill] = {
            "avg": round(avg, 2),
            "count": len(scores),
            "scores": scores,
            "weak": avg < WEAK_SKILL_THRESHOLD,
        }

    return out


def _fallback_learning(skill_scores: dict) -> dict:
    weak = [s for s, v in skill_scores.items() if v["weak"]]
    text = ("Focus on these skills first: " + ", ".join(weak)) if weak else \
           "No weak skills detected. Keep practicing advanced topics."
    return {"text": text, "recommended_topics": weak, "fallback": True}


def _fallback_report(payload: dict) -> dict:
    scores = payload["skill_scores"]
    strengths = [s for s, v in scores.items() if not v["weak"]]
    weaknesses = [s for s, v in scores.items() if v["weak"]]
    rows = "\n".join(f"| {s} | {v['avg']:.1f}/{SCORE_MAX:.0f} |" for s, v in scores.items())
    md = (
        f"# Interview Report - {payload.get('job_title') or ''}\n\n"
        f"## Skill scores\n| Skill | Score |\n|---|---|\n{rows}\n\n"
        f"**Strengths:** {', '.join(strengths) or '-'}\n\n"
        f"**Weaknesses:** {', '.join(weaknesses) or '-'}\n"
    )
    return {"strengths": strengths, "weaknesses": weaknesses, "markdown": md, "fallback": True}


# ---------------------------------------------------------------- nodes ---
def make_nodes(agents: AgentSet) -> dict[str, Callable]:

    def analyze(state: InterviewState) -> dict:
        result, err = _call(
            "analysis", agents.analysis.analyze,
            state["cv_text"], state["job_description_text"],
        )
        if err or not result or not result.get("interview_plan"):
            err = err or {"agent": "analysis", "error": "No interview_plan in result", "attempts": 1}
            return {"status": "failed", "errors": [err]}
        return {
            "analysis": result,
            "plan": result["interview_plan"],
            "history": [],
            "last_score": None,
            "interview_done": False,
            "status": "interviewing",
        }
    def ask_question(state: InterviewState) -> dict:
        plan, history = state["plan"], state.get("history", [])

        update: dict = {}

        # Try to generate a follow-up for the last main question first.
        if history:
            last_item = history[-1]
            last_question = last_item["question"]

            if last_question.get("question_type", "main") == "main":
                follow_up, err = _call(
                    "interview",
                    agents.interview.follow_up_question,
                    last_question,
                    last_item.get("answer", ""),
                    last_item.get("evaluation", {}),
                    history,
                )

                if err:
                    update["errors"] = [err]

                if follow_up is not None:
                    return {
                        **update,
                        "current_question": follow_up,
                        "pending_answer": None,
                    }

        # Follow-ups do not count toward the interview plan.
        if _main_question_count(history) >= plan["total_questions"]:
            return {
                **update,
                "current_question": None,
                "interview_done": True,
            }

        question, err = _call(
            "interview",
            agents.interview.next_question,
            plan,
            history,
            state.get("last_score"),
        )

        if err:
            update["errors"] = update.get("errors", []) + [err]
            question = _fallback_question(plan, history)

        if question is None:
            return {
                **update,
                "current_question": None,
                "interview_done": True,
            }

        return {
            **update,
            "current_question": question,
            "pending_answer": None,
        }

    def await_answer(state: InterviewState) -> dict:
        # interrupt() pauses the graph here until the API resumes it with the
        # candidate's answer. Keep this node free of side effects: on resume
        # LangGraph re-runs it from the top.
        reply = interrupt({"question": state["current_question"]})
        if isinstance(reply, dict) and reply.get("action") == FINISH_ACTION:
            return {"interview_done": True, "pending_answer": None}
        return {"pending_answer": str(reply)}

    def evaluate(state: InterviewState) -> dict:
        question, answer = state["current_question"], state.get("pending_answer") or ""
        ev, err = _call("evaluation", agents.evaluation.evaluate, question, answer)
        update: dict = {}
        if err or not isinstance(ev, dict):
            update["errors"] = [err or {"agent": "evaluation", "error": "bad result", "attempts": 1}]
            ev = {"score": None, "feedback": "Evaluation unavailable.",
                  "correct_points": [], "missing_points": [], "evaluation_failed": True}
        score = ev.get("score")
        return {
            **update,
            "history": [{"question": question, "answer": answer, "evaluation": ev}],
            "last_score": (float(score)if score is not None
            and question.get("question_type", "main") == "main"
            else state.get("last_score")),
            "current_question": None,
            "pending_answer": None,
        }

    def aggregate(state: InterviewState) -> dict:
        return {"skill_scores": aggregate_skill_scores(state.get("history", [])),
                "status": "reporting"}

    def learning(state: InterviewState) -> dict:
        gaps = state["analysis"].get("gap_analysis", {})
        payload = {
            "session_id": state["session_id"],
            "job_title": state["plan"].get("job_title"),
            "skill_scores": state["skill_scores"],
            "gaps": gaps,
            "history": state.get("history", []),
            "report": state["report"],  # the RAG agent evaluates this report
        }
        result, err = _call("learning", agents.learning.recommend, payload)
        if err:
            return {"learning": _fallback_learning(state["skill_scores"]),
                    "errors": [err], "status": "completed"}
        return {"learning": result, "status": "completed"}

    def report(state: InterviewState) -> dict:
        payload = {
            "session_id": state["session_id"],
            "job_title": state["plan"].get("job_title"),
            "plan": state["plan"],
            "analysis": state["analysis"],
            "skill_scores": state["skill_scores"],
            "history": state.get("history", []),
        }
        result, err = _call("report", agents.report.generate, payload)
        if err:
            return {"report": _fallback_report(payload), "errors": [err], "status": "learning"}
        return {"report": result, "status": "learning"}

    return {
        "analyze": analyze, "ask_question": ask_question, "await_answer": await_answer,
        "evaluate": evaluate, "aggregate": aggregate, "report": report, "learning": learning,
    }
