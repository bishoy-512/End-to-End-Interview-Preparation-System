

import argparse
import sys
from pathlib import Path

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True, help="CV file (.pdf/.docx) or .txt")
    ap.add_argument("--jd", required=True, help="Job description .txt file")
    ap.add_argument("--url", default="http://127.0.0.1:8000/interview")
    args = ap.parse_args()

    cv_path, jd = Path(args.cv), Path(args.jd).read_text(encoding="utf-8")
    timeout = httpx.Timeout(600.0)  # real LLM / RAG calls can be slow

    with httpx.Client(timeout=timeout) as http:
        if cv_path.suffix.lower() == ".txt":
            r = http.post(args.url, data={"job_description": jd,
                                          "cv_text": cv_path.read_text(encoding="utf-8")})
        else:
            with cv_path.open("rb") as f:
                r = http.post(args.url, data={"job_description": jd},
                              files={"cv_file": (cv_path.name, f)})

        while True:
            if r.status_code != 200:
                sys.exit(f"Error {r.status_code}: {r.text}")
            data = r.json()

            if data["done"]:
                if data["status"] == "failed":
                    sys.exit(f"Interview failed: {data['errors']}")
                res = data["result"]
                print("\n" + "=" * 60 + "\nREPORT\n" + "=" * 60)
                print(res["report"]["markdown"])
                print("=" * 60 + "\nLEARNING ROADMAP (RAG)\n" + "=" * 60)
                print(res["learning"]["text"])
                if data["errors"]:
                    print("\n[warnings]", data["errors"])
                return

            q, p = data["question"], data["progress"]
            print(f"\n[{p['answered'] + 1}/{p['total']}] ({q['skill']} | {q['difficulty']})")
            print(q["text"])
            answer = input("> ").strip()
            body = {"session_id": data["session_id"]}
            body.update({"finish": "true"} if answer == "/finish" else {"answer": answer})
            r = http.post(args.url, data=body)


if __name__ == "__main__":
    main()
