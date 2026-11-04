"""
Standalone RAG Agent for the Interview System.

This file wraps the existing RAG pipeline as a LangGraph tool-calling agent.
The retrieval pipeline and its prompts are kept the same as in the notebook.

Required project layout:
    project/
    ├── rag_agent.py
    ├── .env
    ├── jobs_info/
    │   ├── AI_ML_Engineer_Knowledge_Base.pdf
    │   └── ... other knowledge-base PDFs
    └── chroma_db/              # created automatically

Required .env:
    GROQ_API_KEY=your_groq_api_key

Run:
    python rag_agent.py

The first run builds the Chroma collection and may take time because embeddings
are generated locally through Ollama. Later runs reuse the persistent collection.
"""

import os
import re
from pathlib import Path
from typing import TypedDict

import chromadb
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


# ============================================================
# 1. Configuration
# ============================================================

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "jobs_info"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "jobs_knowledge"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RAG_LLM_MODEL = "openai/gpt-oss-20b"
AGENT_LLM_MODEL = "openai/gpt-oss-20b"

# Existing RAG LLM client (Groq OpenAI-compatible API).
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# The orchestration LLM belongs to the Agent layer.
# The RAG generation client above remains unchanged.
agent_llm = ChatOpenAI(
    model=AGENT_LLM_MODEL,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0
)


# ============================================================
# 2. Existing PDF processing
# ============================================================

def clean_pdf_text(text):
    # 1. removing header of the pdfs
    text = re.sub(
        r"^[^\n]*Knowledge\s+Base[^\n]*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # 2. removing the footer (page no.)
    text = re.sub(
        r"\bPage\s+\d+\s+of\s+\d+\b",
        "",
        text,
        flags=re.IGNORECASE
    )

    # 3. cleaning up extra spaces and lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def pdf_reader(path):
    paths = sorted(Path(path).glob("*.pdf"))
    documents = []

    for pdf_path in paths:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # 1. Extract normal text and clean it.
                text = clean_pdf_text(page.extract_text() or "")

                # 2. Extract tables, if any.
                tables = page.extract_tables()
                table_texts = []

                for table in tables:
                    formatted_rows = [
                        " | ".join([cell if cell else "" for cell in row])
                        for row in table
                    ]
                    table_texts.append("\n".join(formatted_rows))

                # Merge normal text with extracted tables.
                full_content = text
                if table_texts:
                    full_content += "\n\n--- Tables ---\n\n" + "\n\n".join(table_texts)

                if full_content.strip():
                    documents.append({
                        "title": pdf_path.stem,
                        "page_number": page_num,
                        "text": full_content
                    })

    return documents


# ============================================================
# 3. Existing chunking
# ============================================================

def create_chunks(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = []
    for doc in documents:
        for idx, piece in enumerate(splitter.split_text(doc["text"])):
            chunk = {
                "chunk_id": f"{doc['title']}_{doc['page_number']}_{idx}",
                "title": doc["title"],
                "page_number": doc["page_number"],
                "chunk_idx": idx,
                "text": piece
            }
            chunks.append(chunk)

    return chunks


# ============================================================
# 4. Existing ChromaDB setup
# ============================================================

model_name = EMBEDDING_MODEL
embedder = OllamaEmbeddings(model=model_name)


def chroma_db(chunks, collection_name=COLLECTION_NAME):
    db_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    lc_chroma = Chroma(
        client=db_client,
        collection_name=collection_name,
        embedding_function=embedder
    )

    # Preserve the existing chunk content and metadata structure.
    docs = [
        Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "page_number": chunk["page_number"],
                "chunk_idx": chunk["chunk_idx"]
            }
        )
        for chunk in chunks
    ]

    # Only ingest when the persistent collection is empty.
    # This avoids re-embedding the same PDFs every time the Agent starts.
    if lc_chroma._collection.count() == 0:
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_ids = [
                chunk["chunk_id"]
                for chunk in chunks[i:i + batch_size]
            ]

            lc_chroma.add_documents(batch_docs, ids=batch_ids)
            print(f"Successfully added batch {i} to {i + len(batch_docs)}")

    return lc_chroma


# ============================================================
# 5. Existing retrievers and BM25
# ============================================================

def tokenize(text):
    text = text.lower()
    tokens = re.findall(
        r"[a-z0-9_]+(?:[-][a-z0-9_]+)*",
        text
    )
    return tokens


def build_bm25(chunks):
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]
    return BM25Okapi(tokenized_chunks)


def dense_retriever(query, vector_store, k=10):
    results = vector_store.similarity_search(query, k=k)

    docs = []
    for doc in results:
        metadata = doc.metadata
        docs.append({
            "text": doc.page_content,
            "title": metadata.get("title"),
            "page_number": metadata.get("page_number"),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_idx": metadata.get("chunk_idx")
        })

    return docs


def sparse_retriever(query, bm25, chunks, k=20):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[::-1][:k]
    docs = [chunks[i] for i in top_indices]
    return docs


def rff_combined(ranking_lists, k=60):
    scores = {}
    doc_lookup = {}

    for ranking in ranking_lists:
        for rank, chunk in enumerate(ranking):
            doc_id = (
                chunk.get("chunk_id")
                or chunk.get("page_content")
                or chunk.get("text")
            )
            doc_lookup[doc_id] = chunk
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores.keys(), key=lambda d: -scores[d])
    return [doc_lookup[doc_id] for doc_id in sorted_ids]


def hybrid_retriever(query, bm25, vector_store, chunks, k=10):
    dense_docs = dense_retriever(query, vector_store)
    sparse_docs = sparse_retriever(query, bm25, chunks)

    dense_formatted = [
        {
            "text": doc["text"],
            "title": doc["title"],
            "page_number": doc["page_number"],
            "chunk_id": doc["chunk_id"]
        }
        for doc in dense_docs
    ]

    sparse_formatted = [
        {
            "text": chunk["text"],
            "title": chunk["title"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk.get("chunk_id")
        }
        for chunk in sparse_docs
    ]

    fused_chunks = rff_combined([dense_formatted, sparse_formatted])
    return fused_chunks[:k]


# ============================================================
# 6. Existing CrossEncoder reranker
# ============================================================

print("Loading CrossEncoder reranker...")
reranker = CrossEncoder(RERANKER_MODEL)


# ============================================================
# 7. Existing RAG retrieval + context preparation
# ============================================================

def hybrid_search_context(query, bm25, vector_store, chunks, k=6):

    # Step 1: Hybrid Retrieval
    retrieved_chunks = hybrid_retriever(
        query, bm25, vector_store, chunks, k=k
    )

    if not retrieved_chunks:
        return "", []

    # Step 2: Prepare query-document pairs
    pairs = [
        (query, chunk["text"])
        for chunk in retrieved_chunks
    ]

    # Step 3: Reranking
    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(scores, retrieved_chunks),
        key=lambda item: float(item[0]),
        reverse=True
    )

    reranked_chunks = [
        chunk for score, chunk in ranked_results
    ]

    # Step 4: Build context and source registry
    context_parts = []
    sources = []

    for rank, chunk in enumerate(reranked_chunks, start=1):

        source_id = f"S{rank}"

        source_name = (
            chunk.get("source_file")
            or chunk.get("title")
            or "Unknown document"
        )

        page_number = chunk.get("page_number")
        chunk_id = chunk.get("chunk_id")

        source = {
            "source_id": source_id,
            "document": source_name,
            "page": page_number,
            "chunk_id": chunk_id
        }

        sources.append(source)

        context_parts.append(
            f"""
            [{source_id}]
            Document: {source_name}
            Page: {page_number}
            Chunk ID: {chunk_id}

            Content:
            {chunk["text"]}
            """
        )

    context = "\n\n--------------------\n\n".join(context_parts)

    return context, sources


# ============================================================
# 8. Existing RAG answer generation (prompt preserved)
# ============================================================

def give_advice_with_rag(
    report,
    candidate_query,
    bm25,
    vector_store,
    target_field,
    chunks
):

    # Step 1: Retrieve knowledge for evaluation
    evaluation_context, evaluation_sources = hybrid_search_context(
        candidate_query,
        bm25,
        vector_store,
        chunks,
        k=4
    )

    # Step 2: Retrieve knowledge for learning roadmap
    roadmap_query = f"""
    Learning roadmap for {target_field}.
    Required foundational knowledge, core concepts,
    intermediate and advanced topics, practical skills,
    tools, technologies, prerequisites, and learning progression.
    """

    roadmap_context, roadmap_sources = hybrid_search_context(
        roadmap_query,
        bm25,
        vector_store,
        chunks,
        k=5
    )

    if not evaluation_context and not roadmap_context:
        return "No relevant information found in the Knowledge Base."

    # Step 3: Build the combined context
    combined_context = f"""
    ==============================
    EVALUATION KNOWLEDGE
    ==============================

    {evaluation_context}

    ==============================
    ROADMAP KNOWLEDGE
    ==============================

    {roadmap_context}
    """

    # Combine source information
    all_sources = evaluation_sources + roadmap_sources

    # Remove duplicate sources
    unique_sources = {}
    for source in all_sources:
        unique_sources[source["source_id"]] = source

    source_registry = "\n".join([
        (
            f"[{s['source_id']}] "
            f"Book: {s['document']} | "
            f"Page: {s['page']} | "
            f"Chunk ID: {s['chunk_id']}"
        )
        for s in unique_sources.values()
    ])

    system_prompt = """
        You are a Knowledge-Base-Grounded Interview Evaluator.

        Use ONLY the retrieved context. Do not use external knowledge.

        Rules:
        - Every technical claim and roadmap topic needs a valid citation.
        - Cite using: [Book: title | Page: number | Chunk: id].
        - Never invent facts, topics, or references.
        - If evidence is missing, say: Not found in Knowledge Base.
        - Prioritize the candidate's actual weaknesses.
        - Keep the response concise and avoid repetition.

    """

    user_prompt = f"""
    FIELD: {target_field}

    INTERVIEW REPORT:
    {report}

    EVALUATION CONTEXT:
    {evaluation_context}

    ROADMAP CONTEXT:
    {roadmap_context}

    AVAILABLE SOURCES:
    {source_registry}

    Evaluate the candidate and return:

    1. CORRECTIONS
    - Identify incorrect or incomplete answers.
    - Give the correct explanation based on the context.
    - Cite the supporting source.

    2. PERSONALIZED LEARNING ROADMAP
    Create an ordered roadmap:
    - Start with the candidate's weaknesses.
    - Continue with essential field topics.
    - Progress from fundamentals to advanced topics.
    - For each stage include:
    * Topics
    * Learning objectives
    * Practical tasks, only if supported by context
    * Book title and page citations

    3. CURRENT LEVEL
    - Assess the candidate's level.
    - Mention strengths and weaknesses with evidence.

    4. SOURCES
    - List books and pages actually used.

    5. KNOWLEDGE GAPS
    - Mention topics not sufficiently covered by the retrieved context.

    IMPORTANT:
    Use only retrieved information.
    Every technical correction and roadmap topic must have a citation.
    Do not invent missing information.
    Keep the answer concise but useful.
    """

    # Step 5: Generate response
    response = client.chat.completions.create(
        model=RAG_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=2500
    )

    return response


# ============================================================
# 9. Initialize the existing RAG system
# ============================================================

def initialize_rag():
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge Base folder not found: {KNOWLEDGE_BASE_DIR}"
        )

    pdf_files = list(KNOWLEDGE_BASE_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside: {KNOWLEDGE_BASE_DIR}"
        )

    print("Reading Knowledge Base PDFs...")
    documents = pdf_reader(KNOWLEDGE_BASE_DIR)
    chunks = create_chunks(documents)

    print(f"Loaded {len(documents)} pages and created {len(chunks)} chunks.")

    print("Connecting to ChromaDB...")
    vector_store = chroma_db(chunks)

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    return bm25, vector_store, chunks


# ============================================================
# 10. RAG Tool - the existing RAG is exposed as ONE tool
# ============================================================

def create_rag_tool(bm25, vector_store, chunks):

    @tool
    def evaluate_candidate_with_rag(
        report: str,
        candidate_query: str,
        target_field: str
    ) -> str:
        """
        Use the Knowledge Base RAG system to evaluate an interview report
        and create corrections, a personalized learning roadmap, current
        candidate level, sources, and knowledge gaps.

        Call this tool when the user asks to evaluate interview performance,
        explain weaknesses, or generate a learning roadmap based on a report.

        report: The complete interview evaluation report.
        candidate_query: A focused technical query describing the candidate's
            weaknesses or the concepts that need supporting knowledge.
        target_field: The target job field, for example AI/ML Engineer.
        """
        result = give_advice_with_rag(
            report=report,
            candidate_query=candidate_query,
            bm25=bm25,
            vector_store=vector_store,
            target_field=target_field,
            chunks=chunks
        )

        if isinstance(result, str):
            return result

        return result.choices[0].message.content or ""

    return evaluate_candidate_with_rag


# ============================================================
# 11. Standalone LangGraph Agent + short-term conversation memory
# ============================================================

AGENT_SYSTEM_PROMPT = """
You are the RAG Agent inside an Interview Preparation System.

You have one specialized tool: evaluate_candidate_with_rag.

Rules:
- When the user requests an interview-report evaluation, corrections,
  a personalized learning roadmap, current level, or knowledge gaps,
  you MUST call evaluate_candidate_with_rag.
- Do not independently evaluate technical answers or create a roadmap
  from your own knowledge.
- Pass the complete interview report, a focused candidate query, and
  the target field to the tool.
- After the tool returns, present its result clearly.
- Do not add technical claims that are not present in the tool result.
- If required inputs are missing, ask the user for them.
- For follow-up questions about the returned report, use the conversation
  history and do not fabricate information outside the RAG result.
"""

# In-memory checkpointer: keeps messages for a thread while this Python
# process is running. It is not persistent across application restarts.
memory = MemorySaver()


def build_rag_agent(bm25, vector_store, chunks):
    rag_tool = create_rag_tool(bm25, vector_store, chunks)

    agent = create_react_agent(
        model=agent_llm,
        tools=[rag_tool],
        prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=memory
    )

    return agent


def run_rag_agent(
    agent,
    report: str,
    candidate_query: str,
    target_field: str,
    thread_id: str
):
    """
    Run the standalone RAG Agent.

    Use the same thread_id for the same candidate/session to retain
    short-term conversation context. Use a different thread_id for
    another candidate/session.
    """
    user_message = f"""
Please evaluate this candidate using the RAG Knowledge Base.

TARGET FIELD:
{target_field}

CANDIDATE QUERY:
{candidate_query}

INTERVIEW REPORT:
{report}
"""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"configurable": {"thread_id": thread_id}}
    )

    return result["messages"][-1].content


# ============================================================
# 12. Run standalone from terminal
# ============================================================

def main():
    print("\nInitializing Standalone RAG Agent...")
    bm25, vector_store, chunks = initialize_rag()
    agent = build_rag_agent(bm25, vector_store, chunks)

    print("\nRAG Agent is ready.")
    print("Enter 'exit' to stop.\n")

    while True:
        report = input("\nPaste the interview report (or type exit):\n").strip()
        if report.lower() in {"exit", "quit"}:
            break

        candidate_query = input(
            "\nEnter the candidate weakness/topic query:\n"
        ).strip()

        target_field = input(
            "\nEnter the target field (e.g. AI/ML Engineer):\n"
        ).strip()

        thread_id = input(
            "\nEnter candidate/session ID (same ID keeps conversation memory):\n"
        ).strip()

        if not all([report, candidate_query, target_field, thread_id]):
            print("All fields are required. Please try again.")
            continue

        try:
            answer = run_rag_agent(
                agent=agent,
                report=report,
                candidate_query=candidate_query,
                target_field=target_field,
                thread_id=thread_id
            )
            print("\n" + "=" * 70)
            print("RAG AGENT RESPONSE")
            print("=" * 70)
            print(answer)
            print("=" * 70)

        except Exception as exc:
            print(f"\nAgent execution failed: {exc}")


if __name__ == "__main__":
    main()
