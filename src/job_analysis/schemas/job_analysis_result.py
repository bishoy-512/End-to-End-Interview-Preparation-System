
from pydantic import BaseModel

from schemas.candidate import CandidateProfile
from schemas.job import JobDescription
from schemas.interview_plan_schema import InterviewPlan
from analysis.skill_extractor import SkillMention
from analysis.gap_analyzer import GapAnalysis


class JobAnalysisResult(BaseModel):
    candidate_profile: CandidateProfile
    job_description: JobDescription

    candidate_skills: list[SkillMention]
    job_skills: list[SkillMention]

    gap_analysis: GapAnalysis
    interview_plan: InterviewPlan
