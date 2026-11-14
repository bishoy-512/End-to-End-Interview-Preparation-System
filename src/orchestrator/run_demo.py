
import json
import os

os.environ.setdefault("ORCH_DB", ":memory:")

from langgraph.types import Command  # noqa: E402

from .graph import build_graph  # noqa: E402


def main() -> None:
    g = build_graph()
    cfg = {"configurable": {"thread_id": "demo"}}
    g.invoke({"session_id": "demo", "cv_text": "cv", "job_description_text": "jd"}, cfg)
    while g.get_state(cfg).next:
        q = g.get_state(cfg).values["current_question"]
        print(f"\n[{q['skill']} | {q['difficulty']}] {q['text']}")
        answer = input("> ") or "short answer"
        g.invoke(Command(resume=answer), cfg)
    final = g.get_state(cfg).values
    print("\n" + final["report"]["markdown"])
    print(json.dumps(final["skill_scores"], indent=2))


if __name__ == "__main__":
    main()
