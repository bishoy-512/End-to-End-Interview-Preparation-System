"""
Thin FastAPI layer. Run from the `src/` folder:

    uv run uvicorn orchestrator.api:app --reload

POST /interview                ONE endpoint for the whole flow (use this one):
                                 - first call : job_description + cv (file/text)      -> first question
                                 - next calls : session_id + answer                   -> next question
                                 - last call  : same, returns report + RAG roadmap    -> done=true
                               (add finish=true to end early)

--- lower-level endpoints (same logic, JSON based) ---
POST /sessions                 start: cv (file or text) + job_description -> first question
POST /sessions/{id}/answer     {"answer": "..."} or {"finish": true} -> next question / done
GET  /sessions/{id}            full current state (status, history, scores, errors)
GET  /sessions/{id}/report     final report (409 until the session is completed)
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from langgraph.types import Command
from pydantic import BaseModel

from .agents.real import JOB_ANALYSIS_DIR
from .graph import build_graph

app = FastAPI(title="AI Interview Orchestrator")
_graph = None


def graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _cfg(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _snapshot(session_id: str) -> dict[str, Any]:
    snap = graph().get_state(_cfg(session_id))
    if not snap.values:
        raise HTTPException(404, "Session not found")
    v = snap.values
    plan = v.get("plan") or {}
    history = v.get("history", [])
    waiting = bool(snap.next)  # paused on await_answer
    return {
        "session_id": session_id,
        "status": v.get("status", "created"),
        "question": v.get("current_question") if waiting else None,
        "progress": {"answered": len(history), "total": plan.get("total_questions")},
        "errors": v.get("errors", []),
    }


class AnswerBody(BaseModel):
    answer: str | None = None
    finish: bool = False


def _start(job_description: str, cv_text: str | None, cv_file: UploadFile | None) -> str:
    if not job_description or not job_description.strip():
        raise HTTPException(422, "job_description is required to start an interview")
    if not (cv_text and cv_text.strip()) and cv_file is None:
        raise HTTPException(422, "Provide cv_text or cv_file")

    if cv_file is not None:
        # Reuse Hady's extractor (PDF / DOCX, OCR fallback).
        if str(JOB_ANALYSIS_DIR) not in sys.path:
            sys.path.insert(0, str(JOB_ANALYSIS_DIR))
        from extraction.cv_parser import extract_cv_text

        suffix = Path(cv_file.filename or "cv.pdf").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(cv_file.file.read())
        try:
            cv_text = extract_cv_text(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    session_id = uuid.uuid4().hex
    graph().invoke(
        {"session_id": session_id, "cv_text": cv_text,
         "job_description_text": job_description, "status": "analyzing"},
        _cfg(session_id),
    )
    return session_id


def _resume(session_id: str, answer: str | None, finish: bool) -> None:
    snap = _snapshot(session_id)
    if snap["question"] is None:
        raise HTTPException(409, f"Session is not waiting for an answer (status={snap['status']})")
    if not finish and answer is None:
        raise HTTPException(422, "Provide answer or finish=true")
    resume = {"action": "finish"} if finish else answer
    graph().invoke(Command(resume=resume), _cfg(session_id))


def _final_result(session_id: str) -> dict[str, Any]:
    v = graph().get_state(_cfg(session_id)).values
    return {
        "report": v.get("report"),
        "skill_scores": v.get("skill_scores"),
        "learning": v.get("learning"),  # RAG roadmap (reads the report)
    }


@app.post("/interview")
def interview(
    session_id: str | None = Form(None),
    answer: str | None = Form(None),
    finish: bool = Form(False),
    job_description: str | None = Form(None),
    cv_text: str | None = Form(None),
    cv_file: UploadFile | None = File(None),
):
    """Single dynamic endpoint: start -> answer -> answer -> ... -> final result."""
    if session_id:
        _resume(session_id, answer, finish)
    else:
        session_id = _start(job_description or "", cv_text, cv_file)

    snap = _snapshot(session_id)
    snap["done"] = snap["status"] in ("completed", "failed")
    if snap["status"] == "completed":
        snap["result"] = _final_result(session_id)
    return snap


@app.post("/sessions")
def start_session(
    job_description: str = Form(...),
    cv_text: str | None = Form(None),
    cv_file: UploadFile | None = File(None),
):
    return _snapshot(_start(job_description, cv_text, cv_file))


@app.post("/sessions/{session_id}/answer")
def submit_answer(session_id: str, body: AnswerBody):
    _resume(session_id, body.answer, body.finish)
    return _snapshot(session_id)


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    snap = graph().get_state(_cfg(session_id))
    if not snap.values:
        raise HTTPException(404, "Session not found")
    return {**_snapshot(session_id), "state": snap.values}


@app.get("/sessions/{session_id}/report")
def get_report(session_id: str):
    snap = graph().get_state(_cfg(session_id))
    if not snap.values:
        raise HTTPException(404, "Session not found")
    if snap.values.get("status") != "completed":
        raise HTTPException(409, f"Report not ready (status={snap.values.get('status')})")
    return _final_result(session_id)
