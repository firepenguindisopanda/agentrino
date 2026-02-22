import os
from typing import BinaryIO

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

ALLOWED_EXTENSIONS = {".txt", ".pdf"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)


def validate_file(file: BinaryIO, filename: str) -> tuple[bool, str]:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB"

    return True, ""


def extract_text_from_file(file: BinaryIO, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        return extract_text_from_txt(file)
    elif ext == ".pdf":
        return extract_text_from_pdf(file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def extract_text_from_txt(file: BinaryIO) -> str:
    content = file.read()
    return content.decode("utf-8", errors="ignore")


def extract_text_from_pdf(file: BinaryIO) -> str:
    pdf_reader = PdfReader(file)
    text_parts = []

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text_parts.append(page.extract_text())

    return "\n\n".join(text_parts)


def process_file(file: BinaryIO, filename: str) -> list[Document]:
    is_valid, error = validate_file(file, filename)
    if not is_valid:
        raise ValueError(error)

    text = extract_text_from_file(file, filename)

    if not text.strip():
        raise ValueError("No text content found in file")

    chunks = text_splitter.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        )
        documents.append(doc)

    return documents
