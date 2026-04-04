import json
from datetime import datetime

from src.models import AnalysisReport, RiskLevel


RISK_EMOJI = {
    RiskLevel.High: "🔴",
    RiskLevel.Medium: "🟡",
    RiskLevel.Low: "🟢",
}


def generate_markdown_report(report: AnalysisReport) -> str:
    lines = []

    lines.append(f"# 계약서 법무 검토 리포트")
    lines.append(f"\n> ⚠️ **면책 고지**: {report.disclaimer}")
    lines.append(f"\n---\n")

    lines.append(f"## 기본 정보")
    lines.append(f"- **문서명**: {report.document_name}")
    lines.append(f"- **계약 유형**: {report.contract_type.value}")
    lines.append(f"- **생성 일시**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")

    if report.parties:
        lines.append(f"- **당사자**:")
        for role, name in report.parties.items():
            lines.append(f"  - {role}: {name}")

    high = sum(1 for f in report.findings if f.risk_level == RiskLevel.High)
    mid = sum(1 for f in report.findings if f.risk_level == RiskLevel.Medium)
    low = sum(1 for f in report.findings if f.risk_level == RiskLevel.Low)
    lines.append(f"\n## 리스크 요약")
    lines.append(f"| 레벨 | 건수 |")
    lines.append(f"|------|------|")
    lines.append(f"| 🔴 High | {high} |")
    lines.append(f"| 🟡 Medium | {mid} |")
    lines.append(f"| 🟢 Low | {low} |")
    lines.append(f"| **합계** | **{len(report.findings)}** |")

    lines.append(f"\n---\n")
    lines.append(f"## 위험 조항 상세")

    for i, finding in enumerate(report.findings, 1):
        emoji = RISK_EMOJI.get(finding.risk_level, "⚪")
        lines.append(f"\n### {i}. {emoji} [{finding.risk_level.value}] {finding.clause_category.value}")
        lines.append(f"**요약**: {finding.issue_summary}")
        lines.append(f"\n**근거 (원문 인용)**:")
        lines.append(f"> {finding.evidence}")
        lines.append(f"\n**왜 위험한가**: {finding.why_it_matters}")
        lines.append(f"\n**권장 조치**: `{finding.recommended_action.value}`")
        lines.append(f"\n**신뢰도**: {finding.confidence:.0%}")

        if finding.questions_to_ask:
            lines.append(f"\n**법무 확인 질문**:")
            for q in finding.questions_to_ask:
                lines.append(f"- {q}")

        # RAG 결과
        rag = next((r for r in report.rag_results if r.finding_index == i - 1), None)
        if rag and rag.standard_clauses:
            lines.append(f"\n**📚 표준 조항 비교**:")
            for sc in rag.standard_clauses[:3]:
                lines.append(f"\n> **[{sc.category}] {sc.title}**")
                lines.append(f"> {sc.template_text}")
                if sc.guidance:
                    lines.append(f"> 💡 {sc.guidance}")
            if rag.recommendation:
                lines.append(f"\n**추천 수정안**: {rag.recommendation}")

        # 정책 체크 결과
        pc = next((p for p in report.policy_checks if p.finding_index == i - 1), None)
        if pc:
            conflict_icon = "⚠️ 충돌" if pc.has_conflict else "✅ 이상 없음"
            lines.append(f"\n**🏢 사내 정책 체크** [{pc.policy_name}]: {conflict_icon}")
            if pc.conflict_detail:
                lines.append(f"> {pc.conflict_detail}")
            if pc.recommendation:
                lines.append(f"> 권고: {pc.recommendation}")

        lines.append(f"\n---")

    # 외부 검색 결과
    if report.external_searches:
        lines.append(f"\n## 🌐 외부 규정 검색 결과")
        for es in report.external_searches:
            lines.append(f"\n### 검색: {es.query}")
            lines.append(es.summary)
            if es.sources:
                lines.append(f"\n**출처**:")
                for src in es.sources[:3]:
                    lines.append(f"- [{src.get('title', 'Link')}]({src.get('url', '')})")

    lines.append(f"\n---")
    lines.append(f"\n*이 리포트는 AI 보조 도구로 생성되었으며, 법률 자문이 아닙니다.*")

    return "\n".join(lines)


def generate_json_export(report: AnalysisReport) -> str:
    data = report.model_dump(mode="json")
    # datetime 직렬화
    data["generated_at"] = report.generated_at.isoformat()
    return json.dumps(data, ensure_ascii=False, indent=2)
