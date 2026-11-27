"""Offline tests: no Groq key and no network needed."""

import json
import sys

import pytest

from analysis.gap_analyzer import GapAnalyzer
from analysis.skill_extractor import SkillExtractor, SkillMention
from analysis.skill_normalizer import SkillNormalizer, category_of
from extraction import cv_parser
from planning.interview_planner import InterviewPlanner
from schemas.candidate import CandidateProfile, CandidateSkill
from schemas.job import JobDescription, JobSkill


def cand(name, evidence=None):
    return SkillMention(name=name, source="candidate", evidence=evidence or ["cv"])


def job(name, importance="required"):
    return SkillMention(name=name, source="job", importance=importance, evidence=["jd"])


# ---------------------------------------------------------- normalizer ---
@pytest.mark.parametrize("raw,expected", [
    ("Python programming skills", ["Python"]),
    ("scikit-learn", ["Scikit-learn"]),
    ("Sklearn", ["Scikit-learn"]),
    ("NumPy and Pandas", ["NumPy", "Pandas"]),
    ("Docker and FastAPI", ["Docker", "FastAPI"]),
    ("PyTorch or TensorFlow", ["PyTorch", "TensorFlow"]),
    ("Git & GitHub", ["Git"]),
    ("Machine Learning fundamentals", ["Machine Learning"]),
    ("AWS", ["Cloud Platforms"]),
    ("Retrieval-Augmented Generation (RAG)", ["RAG"]),
    ("Java", ["Java"]),
    ("JavaScript", ["JavaScript"]),
    ("Underwater basket weaving", ["Underwater basket weaving"]),
])
def test_normalize_names(raw, expected):
    assert SkillNormalizer.normalize_skill_names(raw) == expected


def test_normalize_empty():
    assert SkillNormalizer.normalize_skill_names("   ") == []
    assert SkillNormalizer.normalize_skill_name("") == ""


def test_normalize_merges_duplicates_and_keeps_required():
    skills = [
        SkillMention(name="python", source="job", importance="preferred", evidence=["a"]),
        SkillMention(name="Python programming", source="job", importance="required", evidence=["b"]),
    ]
    result = SkillNormalizer.normalize_skills(skills)
    assert len(result) == 1
    assert result[0].name == "Python"
    assert result[0].importance == "required"
    assert result[0].evidence == ["a", "b"]


def test_same_name_different_source_not_merged():
    result = SkillNormalizer.normalize_skills([cand("Python"), job("Python")])
    assert len(result) == 2


def test_category_of():
    assert category_of("Teamwork") == "soft_skill"
    assert category_of("Python") == "programming_language"
    assert category_of("Nonexistent") is None


# ----------------------------------------------------------- gap logic ---
def test_gap_matched_partial_missing_and_score():
    gap = GapAnalyzer.analyze(
        [cand("Python"), cand("TensorFlow")],
        [job("Python"), job("PyTorch"), job("Docker", "preferred")],
    )
    assert [g.skill_name for g in gap.matched_skills] == ["Python"]
    assert [g.skill_name for g in gap.partial_skills] == ["PyTorch"]
    assert gap.partial_skills[0].status == "partial"
    assert [g.skill_name for g in gap.missing_skills] == ["Docker"]
    assert gap.missing_skills[0].status == "not_found_in_cv"
    assert gap.total_required_skills == 2 and gap.total_preferred_skills == 1
    # earned 3 + 1.5 + 0 of 7
    assert gap.match_score == round(100 * 4.5 / 7, 1)


def test_gap_empty_job_skills():
    gap = GapAnalyzer.analyze([cand("Python")], [])
    assert gap.match_score == 0.0


# -------------------------------------------------------------- planner ---
def make_gap(n_required, n_preferred=0, matched=()):
    cands = [cand(s) for s in matched]
    jobs = [job(f"Skill{i}") for i in range(n_required)]
    jobs += [job(f"Extra{i}", "preferred") for i in range(n_preferred)]
    return GapAnalyzer.analyze(cands, jobs)


def test_plan_total_matches_allocation_and_caps_budget():
    plan = InterviewPlanner.create_plan(make_gap(20), "Dev")
    assert plan.total_questions == 12
    assert sum(p.questions_count for p in plan.planned_skills) == 12
    assert len(plan.planned_skills) == 12
    assert len(plan.not_covered) == 8


def test_plan_small_job_uses_whole_budget():
    plan = InterviewPlanner.create_plan(make_gap(3), "Dev", total_questions=9)
    assert plan.total_questions == 9
    assert sum(p.questions_count for p in plan.planned_skills) == 9
    assert all(p.questions_count >= 1 for p in plan.planned_skills)


def test_plan_skips_soft_skills():
    jobs = SkillNormalizer.normalize_skills(
        [job("Python"), job("Teamwork"), job("Communication skills")]
    )
    gap = GapAnalyzer.analyze([], jobs)
    plan = InterviewPlanner.create_plan(gap)
    assert [p.skill_name for p in plan.planned_skills] == ["Python"]


def test_plan_only_soft_skills_raises():
    gap = GapAnalyzer.analyze([], SkillNormalizer.normalize_skills([job("Teamwork")]))
    with pytest.raises(ValueError):
        InterviewPlanner.create_plan(gap)


def test_plan_difficulty_rules():
    gap = GapAnalyzer.analyze(
        [cand("Python", ["a", "b"]), cand("Pandas"), cand("TensorFlow")],
        [job("Python"), job("Pandas", "preferred"), job("PyTorch"), job("Docker")],
    )
    plan = {p.skill_name: p for p in InterviewPlanner.create_plan(gap).planned_skills}
    assert plan["Python"].initial_difficulty == "hard"      # required + strong evidence
    assert plan["Pandas"].initial_difficulty == "medium"
    assert plan["PyTorch"].initial_difficulty == "easy"     # partial
    assert plan["Docker"].initial_difficulty == "easy"      # missing


def test_plan_orders_high_priority_gaps_first():
    gap = GapAnalyzer.analyze(
        [cand("Python")], [job("Python"), job("Docker"), job("NumPy", "preferred")]
    )
    names = [p.skill_name for p in InterviewPlanner.create_plan(gap).planned_skills]
    assert names[0] == "Docker"          # high priority gap before high priority match
    assert names[-1] == "NumPy"


def test_plan_orchestrator_contract():
    """Keys the orchestrator reads: planned_skills[].skill_name/questions_count/
    initial_difficulty, total_questions, job_title."""
    plan = InterviewPlanner.create_plan(make_gap(2), "ML Engineer").model_dump()
    assert plan["job_title"] == "ML Engineer"
    assert isinstance(plan["total_questions"], int)
    for ps in plan["planned_skills"]:
        assert {"skill_name", "questions_count", "initial_difficulty"} <= ps.keys()


# ---------------------------------------------------- tool, end to end ---
def test_tool_pipeline_with_fake_llm(monkeypatch):
    from extraction import jd_parser
    from tools import job_analysis_tool

    profile = CandidateProfile(
        name="Test Candidate",
        skills=[CandidateSkill(name="Python", category="programming_language", evidence=["x"]),
                CandidateSkill(name="TensorFlow", category="framework", evidence=["y"])],
    )
    jd = JobDescription(
        job_title="ML Engineer",
        required_skills=[JobSkill(name="Python programming"), JobSkill(name="PyTorch or Keras")],
        preferred_skills=[JobSkill(name="Docker and FastAPI")],
    )

    def fake(system_prompt, user_prompt, response_schema):
        return profile if response_schema is CandidateProfile else jd

    monkeypatch.setattr(cv_parser, "generate_structured_output", fake)
    monkeypatch.setattr(jd_parser, "generate_structured_output", fake)

    raw = job_analysis_tool.analyze_candidate_for_job.invoke(
        {"cv_text": "cv", "job_description_text": "jd"}
    )
    result = json.loads(raw)
    for key in ("interview_plan", "gap_analysis", "candidate_skills", "job_skills"):
        assert key in result
    assert result["interview_plan"]["job_title"] == "ML Engineer"
    assert result["gap_analysis"]["match_score"] > 0
    names = {s["skill_name"] for s in result["interview_plan"]["planned_skills"]}
    assert {"Python", "PyTorch", "Keras", "Docker", "FastAPI"} == names


def test_tool_returns_error_string_on_empty_input():
    from tools import job_analysis_tool
    out = job_analysis_tool.analyze_candidate_for_job.invoke(
        {"cv_text": " ", "job_description_text": "jd"})
    assert out.startswith("Job Analysis failed")


# ----------------------------------------------------- client and OCR ---
def test_importing_pipeline_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    for mod in [m for m in sys.modules if m.startswith(("services", "tools", "extraction"))]:
        del sys.modules[mod]
    import services.groq_client as gc  # must not raise
    gc.get_client.cache_clear()
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        gc.get_client()


def test_structured_output_retries_and_cleans(monkeypatch):
    from types import SimpleNamespace
    from services import groq_client as gc

    answers = iter(["not json", "```json\n{\"name\": \"A\"}\n```"])

    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(answers)))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    out = gc.generate_structured_output("s", "u", CandidateProfile)
    assert out.name == "A"


def test_structured_output_gives_up(monkeypatch):
    from types import SimpleNamespace
    from services import groq_client as gc

    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    with pytest.raises(ValueError):
        gc.generate_structured_output("s", "u", CandidateProfile)


def test_scanned_pdf_gives_clear_error(tmp_path):
    import pymupdf
    pdf = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    with pytest.raises(ValueError, match="No text found"):
        cv_parser.extract_cv_text(pdf)


def test_unsupported_format(tmp_path):
    f = tmp_path / "cv.txt"
    f.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        cv_parser.extract_cv_text(f)


def test_no_ocr_left():
    assert not hasattr(cv_parser, "extract_ocr_text")
