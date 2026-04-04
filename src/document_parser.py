import hashlib
import json
import re
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def compute_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_cache(file_hash: str) -> dict | None:
    cache_path = CACHE_DIR / f"{file_hash}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return None


def save_cache(file_hash: str, data: dict) -> None:
    cache_path = CACHE_DIR / f"{file_hash}.json"
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".txt":
            return file_bytes.decode("utf-8", errors="replace")
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

        docs: list[Document] = loader.load()
        return "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def split_into_clauses(text: str) -> list[str]:
    """
    조항 번호 패턴(제1조, 1., Article 1, Section 1 등)으로 텍스트를 분리.
    패턴이 없으면 단락 단위로 분리.
    """
    patterns = [
        r"(?=제\s*\d+\s*조)",
        r"(?=\n\s*\d+\.\s)",
        r"(?=\n\s*Article\s+\d+)",
        r"(?=\n\s*Section\s+\d+)",
        r"(?=\n\s*[IVX]+\.\s)",
    ]

    for pattern in patterns:
        parts = re.split(pattern, text)
        parts = [p.strip() for p in parts if len(p.strip()) > 50]
        if len(parts) > 2:
            return parts

    # 패턴 미발견: 단락 단위 분리
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    return paragraphs if paragraphs else [text]


def parse_document(file_bytes: bytes, filename: str) -> dict:
    """
    파일을 파싱하여 ContractDocument 형태의 dict 반환.
    캐시가 있으면 캐시를 반환.
    """
    file_hash = compute_hash(file_bytes)
    cached = load_cache(file_hash)
    if cached:
        cached["from_cache"] = True
        return cached

    raw_text = extract_text_from_bytes(file_bytes, filename)
    clauses = split_into_clauses(raw_text)

    result = {
        "file_name": filename,
        "file_hash": file_hash,
        "raw_text": raw_text,
        "clauses": clauses,
        "from_cache": False,
    }
    save_cache(file_hash, {k: v for k, v in result.items() if k != "from_cache"})
    return result
