"""
contracts/schema.py
───────────────────
Pydantic 도메인 모델 (LCEL 파이프라인 전용).
src/models.py 의 모델을 재사용하면서 분석 상태(AnalysisStatus)와
디버그 컨테이너(AnalysisResult)를 추가합니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# src/models.py 의 기존 Pydantic 모델을 그대로 re-export
from src.models import (
    AnalysisReport,
    ClauseCategory,
    CompanyPolicy,
    ContractAnalysisOutput,
    ContractDocument,
    ContractType,
    ExternalSearchResult,
    PolicyCheckResult,
    RAGResult,
    RecommendedAction,
    RiskFinding,
    RiskLevel,
    StandardClause,
)

__all__ = [
    # re-exported
    "ClauseCategory", "RiskLevel", "RecommendedAction", "ContractType",
    "RiskFinding", "ContractAnalysisOutput", "ContractDocument",
    "StandardClause", "CompanyPolicy", "RAGResult",
    "PolicyCheckResult", "ExternalSearchResult", "AnalysisReport",
    # new
    "AnalysisStatus", "AnalysisResult",
]


class AnalysisStatus(str, Enum):
    SUCCESS     = "success"       # 파싱 성공 + findings 존재
    NO_FINDINGS = "no_findings"   # 파싱 성공 but findings 없음
    PARSE_ERROR = "parse_error"   # LLM 응답은 왔으나 파싱 실패
    LLM_ERROR   = "llm_error"     # LLM 호출 자체 실패


@dataclass
class AnalysisResult:
    """
    단일 계약서 분석의 전체 결과 컨테이너.
    파싱 성공/실패/LLM 오류를 명확히 구분하고
    raw LLM output 및 디버그 정보를 모두 보존합니다.
    """
    status: AnalysisStatus
    output: ContractAnalysisOutput | None = None
    raw_llm_output: str = ""
    parse_error: str = ""
    llm_error: str = ""
    elapsed_sec: float = 0.0
    debug_info: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # ── 편의 프로퍼티 ─────────────────────────────────────────
    @property
    def ok(self) -> bool:
        return self.status in (AnalysisStatus.SUCCESS, AnalysisStatus.NO_FINDINGS)

    @property
    def findings(self):
        return self.output.findings if self.output else []

    @property
    def error_message(self) -> str:
        if self.status == AnalysisStatus.PARSE_ERROR:
            return f"파싱 실패: {self.parse_error}"
        if self.status == AnalysisStatus.LLM_ERROR:
            return f"LLM 오류: {self.llm_error}"
        return ""
