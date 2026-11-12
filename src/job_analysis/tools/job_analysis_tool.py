
from langchain_core.tools import tool

from extraction.cv_parser import CVParser
from extraction.jd_parser import JDParser

from analysis.skill_extractor import SkillExtractor
from analysis.skill_normalizer import SkillNormalizer
from analysis.gap_analyzer import GapAnalyzer

from planning.interview_planner import InterviewPlanner

from schemas.job_analysis_result import JobAnalysisResult


@tool
def analyze_candidate_for_job(
    cv_text: str,
    job_description_text: str
) -> str:
    """
    Analyze a candidate CV against a Job Description.

    This tool performs the complete job analysis pipeline:

    1. Extract candidate information from the CV.
    2. Extract job requirements from the Job Description.
    3. Extract candidate and job skills.
    4. Normalize skill names and remove duplicates.
    5. Perform skill gap analysis.
    6. Generate an interview plan.

    Args:
        cv_text:
            The complete extracted text of the candidate CV.

        job_description_text:
            The complete text of the target Job Description.

    Returns:
        A JSON string containing:
        - Candidate Profile
        - Job Description
        - Candidate Skills
        - Job Skills
        - Gap Analysis
        - Interview Plan
    """

    if not cv_text or not cv_text.strip():
        return "Job Analysis failed: CV text is empty."

    if not job_description_text or not job_description_text.strip():
        return "Job Analysis failed: Job Description is empty."

    try:
        # =========================================
        # 1. CV Parsing
        # =========================================

        candidate_profile = CVParser.parse_text(cv_text)

        # =========================================
        # 2. Job Description Parsing
        # =========================================

        job_description = JDParser.parse_text(
            job_description_text
        )

        # =========================================
        # 3. Skill Extraction
        # =========================================

        candidate_skills = (
            SkillExtractor.extract_candidate_skills(
                candidate_profile
            )
        )

        job_skills = SkillExtractor.extract_job_skills(
            job_description
        )

        # =========================================
        # 4. Skill Normalization
        # =========================================

        candidate_skills = (
            SkillNormalizer.normalize_skills(
                candidate_skills
            )
        )

        job_skills = (
            SkillNormalizer.normalize_skills(
                job_skills
            )
        )

        # =========================================
        # 5. Gap Analysis
        # =========================================

        gap_analysis = GapAnalyzer.analyze(
            candidate_skills=candidate_skills,
            job_skills=job_skills
        )

        # =========================================
        # 6. Interview Planning
        # =========================================

        interview_plan = InterviewPlanner.create_plan(
            gap_analysis=gap_analysis,
            job_title=job_description.job_title
        )

        # =========================================
        # 7. Build Final Output
        # =========================================

        result = JobAnalysisResult(
            candidate_profile=candidate_profile,
            job_description=job_description,
            candidate_skills=candidate_skills,
            job_skills=job_skills,
            gap_analysis=gap_analysis,
            interview_plan=interview_plan
        )

        return result.model_dump_json(
            indent=2,
            exclude_none=False
        )

    except Exception as exc:
        return (
            "Job Analysis execution failed: "
            f"{type(exc).__name__}: {str(exc)}"
        )
