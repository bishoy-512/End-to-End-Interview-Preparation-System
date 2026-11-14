# End-to-End AI Interview Preparation System

An AI-powered interview preparation platform that helps job seekers prepare for real-world interviews through personalized interview questions, interactive mock interviews, AI-powered answer evaluation, and actionable improvement recommendations.

## 💡 Project Overview

The system analyzes job descriptions or job posting URLs to understand job requirements, identify essential skills, and generate personalized interview preparation plans based on the candidate's experience and skill level.

Powered by **Agentic AI** and a multi-agent architecture, the platform provides a complete interview preparation journey, from job analysis to performance evaluation and personalized learning recommendations.

## ✨ Key Features

- **Job Analysis:** Extracts job requirements, responsibilities, qualifications, and essential technical skills from job descriptions.
- **Personalized Interview Questions:** Generates tailored interview questions based on the target role, required skills, and candidate's experience level.
- **Interactive Mock Interviews:** Simulates realistic interview sessions with questions ranging from easy to advanced.
- **AI-Powered Answer Evaluation:** Evaluates candidate answers and provides detailed feedback, including strengths and areas for improvement.
- **Performance Reports:** Generates comprehensive reports highlighting strengths, weaknesses, and skill gaps.
- **Learning Recommendations:** Suggests relevant learning resources to help candidates improve their knowledge and interview performance.

## ⚙️ How It Works

1. **Job Input:** The user provides a job posting URL or pastes a job description.
2. **Job Analysis:** AI agents analyze the job description and extract key requirements, responsibilities, and skills.
3. **Interview Planning:** The system generates personalized interview questions based on the job role and candidate's skill level.
4. **Mock Interview:** The user participates in an interactive interview simulation.
5. **Answer Evaluation:** AI agents evaluate the answers and provide detailed feedback.
6. **Performance Report:** The system generates a final report with improvement recommendations and relevant learning resources.

## 🧠 Agentic AI Architecture

The project follows a **Multi-Agent AI Architecture**, where specialized agents collaborate to handle different stages of the interview preparation process.

Each agent is responsible for a specific task, including:

- **Job Analysis Agent:** Understands job descriptions and extracts relevant requirements and skills.
- **Interview Planning Agent:** Creates personalized interview questions and structures interview sessions.
- **Answer Evaluation Agent:** Analyzes candidate responses and provides detailed feedback.
- **Learning Recommendation Agent:** Identifies knowledge gaps and recommends suitable learning resources.

The agents work together to deliver a personalized and structured interview preparation experience.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Agentic AI Framework:** LangGraph
- **LLM Integration:** OpenAI API
- **Local Model Integration:** Ollama
- **LLM Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Semantic Search:** Sentence Transformers
- **Hybrid Retrieval:** BM25 + Vector Search
- **Reranking:** Cross-Encoder
- **Document Processing:** PDFPlumber
- **Environment & Dependency Management:** uv

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-folder>
```

### 2. Install uv

If uv is not already installed:

```bash
pip install uv
```

Verify the installation:

```bash
uv --version
```

### 3. Create a Virtual Environment

Create a new virtual environment using Python 3.12:

```bash
uv venv --python 3.11
```

### 4. Install Dependencies

Install all project dependencies:

```bash
uv sync
```

This command automatically installs the dependencies defined in `pyproject.toml` and uses `uv.lock` to ensure consistent package versions.

### 5. Configure Environment Variables

Create a `.env` file in the project root directory and add the required configuration:

```env
OPENAI_API_KEY=your_openai_api_key
```

Add any additional environment variables required by your configuration.

### 6. Run the Project

Run the project's main entry point using:

```bash
uv run python main.py
```

Replace `main.py` with the actual entry point if your project uses a different file.

## 📁 Project Structure

```text
project/
│
├── extraction/
│   └── cv_parser.py
│
├── tools/
│   └── job_analysis_tool.py
│
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env
├── .gitignore
└── main.py
```

*Note: The structure above represents the main components and may be extended as the project evolves.*

## 🎯 Project Goal

To make interview preparation more personalized, practical, and accessible by leveraging Agentic AI to help candidates understand job requirements, practice realistic interview scenarios, evaluate their performance, and continuously improve their skills.

---

**Project Track:** Agentic AI

**Initiative:** Digital Egypt Pioneers Initiative (DEPI) — Rwad Misr