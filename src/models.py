from __future__ import annotations
from enum import Enum
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class ClauseCategory(str, Enum):
    Liability = "Liability"
    Termination = "Termination"
    Confidentiality = "Confidentiality"
    IP = "IP"
    Payment = "Payment"
    GoverningLaw = "GoverningLaw"
    Warranty = "Warranty"
    Indemnity = "Indemnity"
    DataPrivacy = "DataPrivacy"
    AutoRenewal = "AutoRenewal"
    Subcontracting = "Subcontracting"
    Other = "Other"


class RiskLevel(str, Enum):
    High = "High"
    Medium = "Medium"
    Low = "Low"


class RecommendedAction(str, Enum):
    수정 = "수정"
    협상 = "협상"
    확인 = "확인"


class ContractType(str, Enum):
    NDA = "NDA"
    Service = "Service"
    License = "License"
    Sales = "Sales"
    Other = "Other"


class RiskFinding(BaseModel):
    clause_category: ClauseCategory = Field(description="조항 카테고리")
    risk_level: RiskLevel = Field(description="리스크 레벨: High/Medium/Low")
    issue_summary: str = Field(description="한 줄 이슈 요약")
    evidence: str = Field(description="원문 인용 + 조항 번호/페이지 위치")
    why_it_matters: str = Field(description="왜 위험한지 설명")
    questions_to_ask: list[str] = Field(description="법무에 확인할 질문 목록")
    recommended_action: RecommendedAction = Field(description="수정/협상/확인 권장 조치")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="추출 신뢰도 0~1")


class ContractAnalysisOutput(BaseModel):
    """LLM 구조화 추출 출력 스키마"""
    contract_type: ContractType = Field(description="계약서 유형")
    parties: dict[str, str] = Field(default_factory=dict, description="계약 당사자 (갑/을 등)")
    findings: list[RiskFinding] = Field(description="발견된 위험 조항 목록")
    disclaimer: str = Field(
        default="이 분석은 법률 자문이 아닌 참고용 보조 도구입니다. 최종 판단은 반드시 법무 전문가에게 확인하세요.",
        description="면책 문구"
    )


class ContractDocument(BaseModel):
    file_name: str
    file_hash: str = Field(description="SHA-256 해시 (캐시 키)")
    raw_text: str
    clauses: list[str] = Field(default_factory=list, description="분리된 조항/섹션 목록")


class StandardClause(BaseModel):
    id: str
    category: str
    title: str
    template_text: str
    guidance: str = ""
    version: str = "1.0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompanyPolicy(BaseModel):
    policy_name: str
    category: str
    rule: str
    examples: list[str] = Field(default_factory=list)


class RAGResult(BaseModel):
    finding_index: int
    standard_clauses: list[StandardClause]
    recommendation: str = ""


class PolicyCheckResult(BaseModel):
    finding_index: int
    policy_name: str
    has_conflict: bool
    conflict_detail: str = ""
    recommendation: str = ""


class ExternalSearchResult(BaseModel):
    query: str
    summary: str
    sources: list[dict[str, str]] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    document_name: str
    contract_type: ContractType = ContractType.Other
    parties: dict[str, str] = Field(default_factory=dict)
    findings: list[RiskFinding] = Field(default_factory=list)
    rag_results: list[RAGResult] = Field(default_factory=list)
    policy_checks: list[PolicyCheckResult] = Field(default_factory=list)
    external_searches: list[ExternalSearchResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
    disclaimer: str = "이 분석은 법률 자문이 아닌 참고용 보조 도구입니다. 최종 판단은 반드시 법무 전문가에게 확인하세요."
