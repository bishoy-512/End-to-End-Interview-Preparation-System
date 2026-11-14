from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[2]
JOB_ANALYSIS_DIR = SRC_DIR / "job_analysis"
RAG_DIR = SRC_DIR / "rag system agent"


def _add_to_path(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


class RealAnalysis:
    """job_analysis.tools.job_analysis_tool.analyze_candidate_for_job"""

    def analyze(self, cv_text: str, job_description_text: str) -> dict[str, Any]:
        _add_to_path(JOB_ANALYSIS_DIR)
        from tools.job_analysis_tool import analyze_candidate_for_job

        raw = analyze_candidate_for_job.invoke(
            {"cv_text": cv_text, "job_description_text": job_description_text}
        )
        # The tool returns a JSON string on success and a plain error string
        # ("Job Analysis failed: ...") on failure -> turn failures into exceptions
        # so the Orchestrator's retry / error handling can see them.
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"Job analysis failed: {raw}") from exc


_rag_lock = threading.Lock()
_rag_agent = None


def _get_rag_agent():
    """Build the RAG agent once (loads Chroma, BM25, CrossEncoder, Ollama)."""
    global _rag_agent
    with _rag_lock:
        if _rag_agent is None:
            _add_to_path(RAG_DIR)
            import os

            # rag_agent.py resolves jobs_info/ and chroma_db/ from its own file
            import rag_agent as rag  # noqa: PLC0415

            bm25, vector_store, chunks = rag.initialize_rag()
            _rag_agent = (rag, rag.build_rag_agent(bm25, vector_store, chunks))
            _ = os  # keep linters quiet
    return _rag_agent


class RealLearning:
    """rag_agent.run_rag_agent(agent, report, candidate_query, target_field, thread_id)"""

    def recommend(self, payload: dict[str, Any]) -> dict[str, Any]:
        rag, agent = _get_rag_agent()

        scores = payload["skill_scores"]
        weak = [s for s, v in scores.items() if v.get("weak")]
        job_title = payload.get("job_title") or "AI/ML Engineer"

        # Plain-text interview report that the RAG agent evaluates:
        # the final report (David) + scores + the raw Q&A.
        lines = [f"Target role: {job_title}", ""]
        final_report = (payload.get("report") or {}).get("markdown")
        if final_report:
            lines += ["Final interview report:", final_report, ""]
        lines += ["Skill scores (0-10):"]
        for skill, v in scores.items():
            lines.append(f"- {skill}: {v['avg']:.1f} ({v['count']} questions)")
        lines += ["", "Answers:"]
        for item in payload.get("history", []):
            q, ev = item["question"], item.get("evaluation", {})
            lines.append(
                f"Q ({q['skill']}, {q['difficulty']}): {q['text']}\n"
                f"A: {item.get('answer', '')}\n"
                f"Score: {ev.get('score')} | Missing: {ev.get('missing_points', [])}"
            )
        report_text = "\n".join(lines)

        query = ", ".join(weak) if weak else job_title
        text = rag.run_rag_agent(
            agent=agent,
            report=report_text,
            candidate_query=query,
            target_field=job_title,
            thread_id=payload.get("session_id", "default"),
        )
        return {"text": text, "recommended_topics": weak}

