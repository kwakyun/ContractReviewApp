import json
from pathlib import Path

from langchain.tools import tool
from tavily import TavilyClient

from src.vectorstore import search_similar_clauses

POLICIES_PATH = Path(__file__).parent.parent / "policies" / "company_policies.json"


def _load_policies() -> list[dict]:
    return json.loads(POLICIES_PATH.read_text(encoding="utf-8"))


@tool
def search_standard_clauses(query: str) -> str:
    """
    계약서 조항 텍스트와 유사한 표준 조항을 ChromaDB에서 검색합니다.
    query: 검색할 계약 조항 텍스트 또는 이슈 요약
    """
    clauses = search_similar_clauses(query, top_k=3)
    if not clauses:
        return "관련 표준 조항을 찾을 수 없습니다."

    results = []
    for i, clause in enumerate(clauses, 1):
        results.append(
            f"[{i}] {clause.title} (카테고리: {clause.category})\n"
            f"표준 문구: {clause.template_text}\n"
            f"가이드: {clause.guidance}"
        )
    return "\n\n".join(results)


@tool
def check_company_policy(category: str) -> str:
    """
    사내 정책 JSON 파일에서 특정 카테고리의 정책을 조회합니다.
    category: 조회할 정책 카테고리 (예: Liability, Confidentiality, GoverningLaw, Payment, IP, DataPrivacy, AutoRenewal, Subcontracting)
    """
    policies = _load_policies()
    matched = [p for p in policies if p["category"].lower() == category.lower()]

    if not matched:
        return f"'{category}' 카테고리에 해당하는 사내 정책이 없습니다."

    results = []
    for p in matched:
        examples = "\n  - ".join(p.get("examples", []))
        results.append(
            f"[정책명] {p['policy_name']}\n"
            f"[규칙] {p['rule']}\n"
            f"[예시]\n  - {examples}"
        )
    return "\n\n".join(results)


@tool
def search_external_regulations(query: str) -> str:
    """
    Tavily를 사용하여 최신 법령/규정/가이드라인을 외부에서 검색합니다.
    query: 검색할 규정 관련 질문 (예: '개인정보 국외이전 최신 규정 2024')
    """
    import os
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY가 설정되지 않아 외부 검색을 수행할 수 없습니다."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )

        answer = response.get("answer", "")
        sources = response.get("results", [])

        output_parts = []
        if answer:
            output_parts.append(f"[요약]\n{answer}")

        if sources:
            source_lines = []
            for s in sources[:3]:
                title = s.get("title", "")
                url = s.get("url", "")
                snippet = s.get("content", "")[:300]
                source_lines.append(f"- {title}\n  URL: {url}\n  내용: {snippet}...")
            output_parts.append("[출처]\n" + "\n".join(source_lines))

        return "\n\n".join(output_parts) if output_parts else "검색 결과가 없습니다."
    except Exception as e:
        return f"외부 검색 중 오류가 발생했습니다: {str(e)}"


ALL_TOOLS = [search_standard_clauses, check_company_policy, search_external_regulations]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
