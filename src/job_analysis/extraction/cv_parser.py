
from pathlib import Path

import pymupdf
from docx import Document

from schemas.candidate import CandidateProfile
from services.groq_client import generate_structured_output
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "cv_extraction.txt"
)


# --------------------------------------------------
# 1. PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(file_path: Path) -> str:
    """Extract text from a PDF file."""

    text_parts = []

    with pymupdf.open(file_path) as pdf:
        for page in pdf:
            text_parts.append(page.get_text("text"))

    return "\n".join(text_parts).strip()


# --------------------------------------------------
# 2. DOCX TEXT EXTRACTION
# --------------------------------------------------

def extract_docx_text(file_path: Path) -> str:
    """Extract text from a DOCX file."""

    document = Document(file_path)

    text_parts = []

    # Extract paragraphs
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)

    # Extract tables
    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip()
                for cell in row.cells
            )

            if row_text.strip():
                text_parts.append(row_text)

    return "\n".join(text_parts).strip()


# --------------------------------------------------
# 3. OCR EXTRACTION FOR SCANNED PDFS
# --------------------------------------------------

def extract_ocr_text(file_path: Path) -> str:
    """Extract text from scanned PDF pages using OCR."""

    from PIL import Image

    text_parts = []

    with pymupdf.open(file_path) as pdf:
        for page in pdf:

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False
            )

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            page_text = pytesseract.image_to_string(
                image,
                lang="eng+ara"
            )

            text_parts.append(page_text)

    return "\n".join(text_parts).strip()


# --------------------------------------------------
# 4. MAIN TEXT EXTRACTION FUNCTION
# --------------------------------------------------

def extract_cv_text(file_path: str | Path) -> str:
    """
    Extract CV text based on its file extension.

    Supported formats:
        - PDF
        - DOCX

    Scanned PDFs use OCR as a fallback.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"CV file not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        text = extract_pdf_text(file_path)

        # Use OCR if the PDF contains no meaningful text
        if len(text.strip()) < 50:
            text = extract_ocr_text(file_path)

    elif extension == ".docx":

        text = extract_docx_text(file_path)

    else:
        raise ValueError(
            "Unsupported CV format. Use PDF or DOCX."
        )

    if not text.strip():
        raise ValueError(
            "Could not extract readable text from the CV."
        )

    return text


# --------------------------------------------------
# 5. LOAD EXTRACTION PROMPT
# --------------------------------------------------

def load_prompt() -> str:

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"CV extraction prompt not found: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(encoding="utf-8")


# --------------------------------------------------
# 6. CV PARSER
# --------------------------------------------------

class CVParser:
    
    @staticmethod
    def parse(file_path: str | Path) -> CandidateProfile:
        cv_text = extract_cv_text(file_path)
        return CVParser.parse_text(cv_text)

    @staticmethod
    def parse_text(cv_text: str) -> CandidateProfile:
        if not cv_text or not cv_text.strip():
            raise ValueError("CV text is empty.")

        prompt = load_prompt()

        return generate_structured_output(
            system_prompt=prompt,
            user_prompt=f"Extract the candidate information from this CV:\n\n{cv_text}",
            response_schema=CandidateProfile
        )
