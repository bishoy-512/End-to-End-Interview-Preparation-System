
import json
from pathlib import Path

from extraction.cv_parser import extract_cv_text
from tools.job_analysis_tool import analyze_candidate_for_job


# =========================================
# 1. Your CV File
# =========================================

CV_FILE_PATH = Path(
    r"path for your CV file"
)


# =========================================
# 2. Sample ML Engineer Job Description
# =========================================

ML_ENGINEER_JD = """
Job Title: Machine Learning Engineer

Company: AI Solutions

Location: Cairo, Egypt

Employment Type: Full-time

Job Summary:

We are looking for a Machine Learning Engineer
to develop, train, evaluate, and deploy machine
learning models for real-world applications.

Responsibilities:

- Build and train machine learning models.
- Perform data preprocessing and feature engineering.
- Evaluate models using appropriate performance metrics.
- Develop deep learning solutions when required.
- Collaborate with software engineers and data teams.
- Optimize model performance.
- Work with NLP and generative AI applications.

Required Skills:

- Strong Python programming skills.
- Machine Learning fundamentals.
- Data preprocessing and feature engineering.
- Supervised and Unsupervised Learning.
- Model evaluation and validation.
- Scikit-learn.
- NumPy and Pandas.

Preferred Skills:

- Deep Learning.
- PyTorch or TensorFlow.
- Natural Language Processing.
- Large Language Models.
- Retrieval-Augmented Generation (RAG).
- Docker and FastAPI.
- Experience with cloud platforms.

Required Education:

- Bachelor's degree in Computer Science,
  Artificial Intelligence, or a related field.

Required Experience:

- No professional experience is required
  for candidates with strong practical projects.

Qualifications:

- Analytical thinking.
- Problem-solving skills.
- Ability to work in a team.
- Good communication skills.

Technologies:

Python, Scikit-learn, PyTorch, TensorFlow,
Pandas, NumPy, FastAPI, Docker.

Eligibility Requirements:

- Ability to work full-time in Cairo.
"""


# =========================================
# 3. Run Job Analysis Tool
# =========================================

def main():

    if not CV_FILE_PATH.exists():
        raise FileNotFoundError(
            f"CV file not found: {CV_FILE_PATH}"
        )

    print("Extracting CV text...")

    cv_text = extract_cv_text(CV_FILE_PATH)

    print(f"CV extracted successfully: {len(cv_text)} characters")

    print("\nRunning Job Analysis Tool...")

    result = analyze_candidate_for_job.invoke(
        {
            "cv_text": cv_text,
            "job_description_text": ML_ENGINEER_JD
        }
    )

    # =====================================
    # 4. Parse and Display Result
    # =====================================

    if not isinstance(result, str):
        raise TypeError(
            f"Unexpected tool output: {type(result)}"
        )

    try:
        result_data = json.loads(result)

    except json.JSONDecodeError:
        print("\nTool returned an error:")
        print(result)
        return

    print("\n" + "=" * 70)
    print("JOB ANALYSIS TOOL OUTPUT")
    print("=" * 70)

    print(json.dumps(
        result_data,
        indent=4,
        ensure_ascii=False
    ))

    # =====================================
    # 5. Save Output
    # =====================================

    output_path = Path("sample_data/job_analysis_result.json")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            result_data,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print(f"Result saved to: {output_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
