import os
import sys
from pathlib import Path

os.environ["ORCH_DB"] = ":memory:"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from fastapi.testclient import TestClient
from langgraph.types import Command

from orchestrator.agents import AgentSet
from orchestrator.agents import mocks
from orchestrator import nodes
from orchestrator.graph import build_graph

nodes.RETRY_DELAY_S = 0  # no sleeping in tests


def mock_agents(**over) -> AgentSet:
    base = dict(analysis=mocks.MockAnalysis(), interview=mocks.MockInterview(),
                evaluation=mocks.MockEvaluation(), learning=mocks.MockLearning(),
                report=mocks.MockReport())
    base.update(over)
    return AgentSet(**base)


def run(graph, tid="t", answer="a reasonably long answer about the topic at hand"):
    cfg = {"configurable": {"thread_id": tid}}
    graph.invoke({"session_id": tid, "cv_text": "cv", "job_description_text": "jd"}, cfg)
    n = 0
    while graph.get_state(cfg).next:
        graph.invoke(Command(resume=answer), cfg)
        n += 1
    return graph.get_state(cfg).values, n


def test_full_flow_with_mocks():
    state, n = run(build_graph(mock_agents()))

    main_questions = [
        item for item in state["history"]
        if item["question"].get("question_type", "main") == "main"
    ]

    follow_up_questions = [
        item for item in state["history"]
        if item["question"].get("question_type") == "follow_up"
    ]

    assert state["status"] == "completed"
    assert len(main_questions) == state["plan"]["total_questions"] == 8
    assert len(state["history"]) == n
    assert len(follow_up_questions) <= 8
    assert set(state["skill_scores"]) == {"Python", "Machine Learning", "Docker"}
    assert "markdown" in state["report"]
    assert state["errors"] == []


def test_mock_plan_matches_hady_schema():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "job_analysis"))
    try:
        from schemas.interview_plan_schema import InterviewPlan
    except ImportError:
        pytest.skip("job_analysis not next to orchestrator")
    InterviewPlan.model_validate(mocks.MockAnalysis().analyze("", "")["interview_plan"])


def test_early_finish_still_produces_report():
    g = build_graph(mock_agents())
    cfg = {"configurable": {"thread_id": "e"}}
    g.invoke({"session_id": "e", "cv_text": "cv", "job_description_text": "jd"}, cfg)
    g.invoke(Command(resume="one answer"), cfg)
    g.invoke(Command(resume={"action": "finish"}), cfg)
    s = g.get_state(cfg).values
    assert s["status"] == "completed" and len(s["history"]) == 1


class Flaky:
    def __init__(self, fail_times): self.n = 0; self.fail_times = fail_times


class FlakyEval(Flaky):
    def evaluate(self, q, a):
        self.n += 1
        if self.n <= self.fail_times:
            raise TimeoutError("llm down")
        return mocks.MockEvaluation().evaluate(q, a)


def test_retry_recovers_transient_failure():
    ev = FlakyEval(fail_times=2)  # fails twice, succeeds on 3rd attempt
    state, _ = run(build_graph(mock_agents(evaluation=ev)))
    assert state["status"] == "completed"
    assert state["history"][0]["evaluation"]["score"] is not None
    assert state["errors"] == []


def test_evaluation_total_failure_falls_back():
    state, _ = run(build_graph(mock_agents(evaluation=FlakyEval(fail_times=10**6))), "f")
    assert state["status"] == "completed"
    assert all(h["evaluation"]["evaluation_failed"] for h in state["history"])
    assert state["errors"] and state["errors"][0]["agent"] == "evaluation"


class BrokenInterview:
    def next_question(self, *a): raise RuntimeError("boom")
    def follow_up_question(self, *a):
        return None

def test_interview_failure_uses_fallback_questions():
    state, n = run(build_graph(mock_agents(interview=BrokenInterview())), "i")
    assert state["status"] == "completed"
    assert n == state["plan"]["total_questions"]
    assert all(h["question"]["id"].startswith("fallback") for h in state["history"])


class BrokenAnalysis:
    def analyze(self, *a): raise RuntimeError("groq down")


def test_analysis_failure_marks_session_failed():
    g = build_graph(mock_agents(analysis=BrokenAnalysis()))
    cfg = {"configurable": {"thread_id": "x"}}
    g.invoke({"session_id": "x", "cv_text": "cv", "job_description_text": "jd"}, cfg)
    s = g.get_state(cfg)
    assert s.values["status"] == "failed" and not s.next


def test_report_and_learning_failures_fall_back():
    class Bad:
        def recommend(self, p): raise RuntimeError("x")
        def generate(self, p): raise RuntimeError("x")
    state, _ = run(build_graph(mock_agents(learning=Bad(), report=Bad())), "r")
    assert state["status"] == "completed"
    assert state["report"]["fallback"] and state["learning"]["fallback"]


def test_adaptive_difficulty_uses_last_score():
    plan = mocks.MockAnalysis().analyze("", "")["interview_plan"]
    h = [{"question": {"skill": "Python"}}]
    easy = mocks.MockInterview().next_question(plan, h, 2.0)
    hard = mocks.MockInterview().next_question(plan, h, 9.0)
    assert easy["difficulty"] == "easy" and hard["difficulty"] == "hard"


def test_api_flow():
    from orchestrator import api
    api._graph = build_graph(mock_agents())
    c = TestClient(api.app)
    r = c.post("/sessions", data={"job_description": "jd", "cv_text": "cv"}).json()
    sid = r["session_id"]
    assert r["question"] and r["progress"]["total"] == 8
    assert c.get(f"/sessions/{sid}/report").status_code == 409
    while r["question"]:
        r = c.post(f"/sessions/{sid}/answer", json={"answer": "some answer here"}).json()
    assert r["status"] == "completed"
    assert c.post(f"/sessions/{sid}/answer", json={"answer": "x"}).status_code == 409
    rep = c.get(f"/sessions/{sid}/report").json()
    assert rep["report"]["markdown"]
    assert c.get("/sessions/nope").status_code == 404


def test_single_interview_endpoint_end_to_end():
    from orchestrator import api

    api._graph = build_graph(mock_agents())
    c = TestClient(api.app)

    r = c.post(
        "/interview",
        data={"job_description": "jd", "cv_text": "cv"}
    ).json()

    sid, steps = r["session_id"], 0
    assert r["done"] is False and r["question"]

    while not r["done"]:
        r = c.post(
            "/interview",
            data={"session_id": sid, "answer": "a decent answer here"}
        ).json()
        steps += 1

    assert steps >= 8
    assert r["status"] == "completed"
    assert r["question"] is None
    assert r["result"]["report"]["markdown"]
    assert r["result"]["learning"]["text"]

    assert c.post(
        "/interview",
        data={"job_description": "jd"}
    ).status_code == 422  # no CV

    r2 = c.post(
        "/interview",
        data={"job_description": "jd", "cv_text": "cv"}
    ).json()

    r2 = c.post(
        "/interview",
        data={"session_id": r2["session_id"], "finish": "true"}
    ).json()

    assert r2["done"] and r2["result"]["report"]


def test_learning_runs_after_report_and_receives_it():
    seen = {}
    class SpyLearning:
        def recommend(self, p):
            seen["report"] = p["report"]
            return {"text": "t", "recommended_topics": []}
    state, _ = run(build_graph(mock_agents(learning=SpyLearning())), "o")
    assert seen["report"]["markdown"] == state["report"]["markdown"]


def test_llm_interview_writes_text_and_follows_plan():
    from orchestrator.agents.llm_interview import LLMInterview

    prompts = []

    def fake(system, user, schema):
        prompts.append(user)
        return schema(text=f"  Real question #{len(prompts)}?  ")

    state, n = run(
        build_graph(mock_agents(interview=LLMInterview(generate=fake))),
        "l",
    )

    main_questions = [
        h for h in state["history"]
        if h["question"].get("question_type", "main") == "main"
    ]

    follow_up_questions = [
        h for h in state["history"]
        if h["question"].get("question_type") == "follow_up"
    ]

    assert state["status"] == "completed"
    assert len(main_questions) == state["plan"]["total_questions"] == 8
    assert len(state["history"]) == n
    assert len(follow_up_questions) <= 8

    assert main_questions[0]["question"]["text"] == "Real question #1?"
    assert main_questions[0]["question"]["question_type"] == "main"
    assert main_questions[0]["question"]["parent_question_id"] is None
    assert "Real question #1?" in prompts[1]

    assert [
        h["question"]["skill"] for h in main_questions
    ].count("Python") == 3


def test_adaptive_difficulty_is_cumulative():
    plan = mocks.MockAnalysis().analyze("", "")["interview_plan"]

    history = [
        {
            "question": {
                "skill": "Python",
                "difficulty": "medium"
            },
            "evaluation": {"score": 9}
        },
        {
            "question": {
                "skill": "Python",
                "difficulty": "hard"
            },
            "evaluation": {"score": 9}
        }
    ]

    question = mocks.MockInterview().next_question(
        plan, history, 9
    )

    assert question["difficulty"] == "hard"


def test_adaptive_difficulty_decreases_cumulatively():
    plan = mocks.MockAnalysis().analyze("", "")["interview_plan"]

    history = [
        {
            "question": {
                "skill": "Python",
                "difficulty": "medium"
            },
            "evaluation": {"score": 2}
        },
        {
            "question": {
                "skill": "Python",
                "difficulty": "easy"
            },
            "evaluation": {"score": 2}
        }
    ]

    question = mocks.MockInterview().next_question(
        plan, history, 2
    )

    assert question["difficulty"] == "easy"


def test_llm_interview_retries_when_question_is_duplicate():
    from orchestrator.agents.llm_interview import LLMInterview

    plan = mocks.MockAnalysis().analyze("", "")["interview_plan"]

    history = [
        {
            "question": {
                "skill": "Python",
                "text": "What is Python?"
            },
            "evaluation": {"score": 7}
        }
    ]

    generated_questions = [
        "  WHAT IS PYTHON?!  ",
        "What are Python decorators?"
    ]

    calls = []

    def fake_generate(system, user, schema):
        calls.append(user)
        return schema(text=generated_questions[len(calls) - 1])

    agent = LLMInterview(generate=fake_generate)

    result = agent.next_question(plan, history, None)

    assert len(calls) == 2
    assert result["text"] == "What are Python decorators?"
    assert result["skill"] == "Python"



def test_llm_interview_raises_error_after_three_duplicate_attempts():
    from orchestrator.agents.llm_interview import LLMInterview

    plan = mocks.MockAnalysis().analyze("", "")["interview_plan"]

    history = [
        {
            "question": {
                "skill": "Python",
                "text": "What is Python?"
            },
            "evaluation": {"score": 7}
        }
    ]

    calls = []

    def fake_generate(system, user, schema):
        calls.append(user)
        return schema(text="What is Python?!")

    agent = LLMInterview(generate=fake_generate)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        agent.next_question(plan, history, None)

    assert len(calls) == 3


def test_follow_up_score_does_not_update_last_score():
    class DifferentScores:
        def evaluate(self, question, answer):
            if question.get("question_type") == "follow_up":
                return {
                    "score": 2.0,
                    "feedback": "Follow-up feedback",
                    "correct_points": [],
                    "missing_points": ["Basic concept"],
                }

            return {
                "score": 8.0,
                "feedback": "Main question feedback",
                "correct_points": ["Correct"],
                "missing_points": ["Additional detail"],
            }

    state, _ = run(
        build_graph(mock_agents(evaluation=DifferentScores())),
        "follow-up-score-test",
    )

    follow_up_items = [
        item for item in state["history"]
        if item["question"].get("question_type") == "follow_up"
    ]

    assert follow_up_items
    assert follow_up_items[0]["evaluation"]["score"] == 2.0
    assert state["last_score"] == 8.0

def test_llm_interview_uses_retrieved_context():
    from orchestrator.agents.llm_interview import LLMInterview

    plan = mocks.MockAnalysis().analyze("", "")["interview_plan"]
    retrieved_contexts = []
    prompts = []

    def fake_retrieve(query):
        retrieved_contexts.append(query)
        return "Python context: decorators and generators."

    def fake_generate(system, user, schema):
        prompts.append(user)
        return schema(text="What are Python decorators?")

    agent = LLMInterview(
        generate=fake_generate,
        retrieve_context=fake_retrieve,
    )

    result = agent.next_question(plan, [], None)

    assert result is not None
    assert len(retrieved_contexts) == 1
    assert "Skill: Python" in retrieved_contexts[0]
    assert "Python context: decorators and generators." in prompts[0]