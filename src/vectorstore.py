import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.models import StandardClause

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "standard_clauses"
STANDARD_CLAUSES_PATH = Path(__file__).parent.parent / "data" / "standard_clauses.json"


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_standard_clauses() -> list[StandardClause]:
    data = json.loads(STANDARD_CLAUSES_PATH.read_text(encoding="utf-8"))
    return [StandardClause(**item) for item in data]


def seed_vectorstore(force: bool = False) -> None:
    """표준 조항 라이브러리를 ChromaDB에 임베딩하여 저장."""
    vs = get_vectorstore()

    if not force and vs._collection.count() > 0:
        return  # 이미 시드됨

    if force:
        vs._collection.delete(where={"id": {"$ne": ""}})

    clauses = load_standard_clauses()
    texts = [f"{c.title}\n\n{c.template_text}\n\n{c.guidance}" for c in clauses]
    metadatas = [
        {
            "id": c.id,
            "category": c.category,
            "title": c.title,
            "template_text": c.template_text,
            "guidance": c.guidance,
            "version": c.version,
        }
        for c in clauses
    ]
    ids = [c.id for c in clauses]

    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def search_similar_clauses(query: str, top_k: int = 3) -> list[StandardClause]:
    """쿼리와 유사한 표준 조항 Top-k를 반환."""
    vs = get_vectorstore()

    if vs._collection.count() == 0:
        seed_vectorstore()

    docs = vs.similarity_search(query, k=top_k)

    clauses: list[StandardClause] = []
    for doc in docs:
        meta = doc.metadata
        clauses.append(
            StandardClause(
                id=meta.get("id", ""),
                category=meta.get("category", ""),
                title=meta.get("title", ""),
                template_text=meta.get("template_text", ""),
                guidance=meta.get("guidance", ""),
                version=meta.get("version", "1.0"),
            )
        )
    return clauses


def add_clause_to_vectorstore(clause: StandardClause) -> None:
    """새로운 표준 조항을 VectorDB에 추가."""
    vs = get_vectorstore()
    text = f"{clause.title}\n\n{clause.template_text}\n\n{clause.guidance}"
    vs.add_texts(
        texts=[text],
        metadatas=[{
            "id": clause.id,
            "category": clause.category,
            "title": clause.title,
            "template_text": clause.template_text,
            "guidance": clause.guidance,
            "version": clause.version,
        }],
        ids=[clause.id],
    )
