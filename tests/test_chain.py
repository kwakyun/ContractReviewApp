"""
tests/test_chain.py
────────────────────
ContractAnalysisChain LCEL 파이프라인 테스트.

실행:
    py -m pytest tests/ -v
    py -m pytest tests/ -v -k "not live"   # LLM 호출 없이 단위 테스트만
"""
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.chain import ContractAnalysisChain  # noqa: E402
from contracts.schema import AnalysisResult, AnalysisStatus, ContractAnalysisOutput  # noqa: E402


# ── 픽스처 ────────────────────────────────────────────────────────
SAMPLE_CONTRACT_KO = """
제1조 (목적)
본 계약은 갑(발주자)과 을(수행사) 사이의 소프트웨어 개발 용역에 관한 사항을 정함을 목적으로 한다.

제5조 (책임 한도)
을의 손해배상 책임은 어떠한 경우에도 한도 없이 무제한으로 적용된다.
고의·과실 여부를 불문하고 을은 모든 손해에 대해 배상하여야 한다.

제8조 (비밀유지)
비밀유지 기간은 계약 종료 후 1개월로 한다.
을은 갑의 영업비밀을 임의로 제3자에 제공할 수 있다.

제12조 (준거법 및 관할)
본 계약은 미국 캘리포니아 주 법률에 따라 규율되며, 분쟁 발생 시 미국 법원을 전속 관할로 한다.
"""

SAMPLE_CONTRACT_EN = """
ARTICLE 5 – LIMITATION OF LIABILITY
Vendor's total liability shall not exceed the greater of (i) USD 1,000 or
(ii) the fees paid by Client in the preceding 3 months.
In no event shall either party be liable for indirect, incidental or
consequential damages.

ARTICLE 8 – CONFIDENTIALITY
Each party shall maintain the confidentiality of the other party's
Confidential Information for a period of 3 years following disclosure.

ARTICLE 12 – GOVERNING LAW
This Agreement shall be governed by the laws of Korea.
"""


def make_mock_chain() -> ContractAnalysisChain:
    """실제 LLM 없이 테스트하기 위한 mock 체인."""
    mock_llm = MagicMock()
    chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
    chain.llm = mock_llm
    chain.debug = True
    return chain


# ══════════════════════════════════════════════════════════════════
# 1. 스키마 / 상태 테스트  (LLM 호출 없음)
# ══════════════════════════════════════════════════════════════════
class TestAnalysisResult:
    def test_ok_property_success(self):
        r = AnalysisResult(status=AnalysisStatus.SUCCESS)
        assert r.ok is True

    def test_ok_property_no_findings(self):
        r = AnalysisResult(status=AnalysisStatus.NO_FINDINGS)
        assert r.ok is True

    def test_not_ok_parse_error(self):
        r = AnalysisResult(status=AnalysisStatus.PARSE_ERROR, parse_error="bad json")
        assert r.ok is False
        assert "파싱 실패" in r.error_message

    def test_not_ok_llm_error(self):
        r = AnalysisResult(status=AnalysisStatus.LLM_ERROR, llm_error="timeout")
        assert r.ok is False
        assert "LLM 오류" in r.error_message

    def test_findings_empty_when_no_output(self):
        r = AnalysisResult(status=AnalysisStatus.PARSE_ERROR)
        assert r.findings == []


class TestContractAnalysisOutput:
    def test_valid_minimal(self):
        out = ContractAnalysisOutput(
            contract_type="Service",
            parties={},
            findings=[],
        )
        assert out.contract_type.value == "Service"
        assert out.findings == []

    def test_disclaimer_default(self):
        out = ContractAnalysisOutput(contract_type="NDA", parties={}, findings=[])
        assert "법률 자문" in out.disclaimer


# ══════════════════════════════════════════════════════════════════
# 2. 체인 입력 빌더 테스트  (LLM 없음)
# ══════════════════════════════════════════════════════════════════
class TestChainInput:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.chain = ContractAnalysisChain.build(debug=True)

    def test_input_contains_format_instructions(self):
        inp = self.chain._inp("sample text")
        assert "format_instructions" in inp
        assert len(inp["format_instructions"]) > 50

    def test_input_truncates_long_text(self):
        long_text = "A" * 20_000
        inp = self.chain._inp(long_text)
        assert len(inp["contract_text"]) <= 12_000

    def test_parser_format_instructions_has_schema(self):
        instructions = self.chain.parser.get_format_instructions()
        assert "contract_type" in instructions
        assert "risk_level" in instructions
        assert "findings" in instructions


# ══════════════════════════════════════════════════════════════════
# 3. 실제 LLM 호출 테스트  (마커: live)
#    실행: py -m pytest tests/ -v -m live
# ══════════════════════════════════════════════════════════════════
@pytest.mark.live
class TestLiveChain:
    """실제 OpenAI API 호출. OPENAI_API_KEY 필요."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        self.chain = ContractAnalysisChain.build(debug=True)

    # ── invoke ─────────────────────────────────────────────────
    def test_invoke_korean_contract(self):
        result = self.chain.invoke(SAMPLE_CONTRACT_KO)
        assert isinstance(result, AnalysisResult)
        assert result.ok, f"분석 실패: {result.error_message}\nRAW: {result.raw_llm_output[:500]}"
        assert len(result.findings) > 0, "위험 조항이 탐지되지 않았습니다."

    def test_invoke_english_contract(self):
        result = self.chain.invoke(SAMPLE_CONTRACT_EN)
        assert result.ok
        assert result.output.contract_type is not None

    def test_invoke_has_raw_output(self):
        result = self.chain.invoke(SAMPLE_CONTRACT_KO)
        assert len(result.raw_llm_output) > 10, "RAW LLM 출력이 비어 있습니다."

    def test_invoke_elapsed_time_recorded(self):
        result = self.chain.invoke(SAMPLE_CONTRACT_KO)
        assert result.elapsed_sec > 0

    def test_parse_error_vs_no_findings(self):
        """빈 텍스트 → NO_FINDINGS 또는 PARSE_ERROR (둘 다 허용, SUCCESS만 금지)."""
        result = self.chain.invoke("이 문서에는 아무 계약 내용이 없습니다.")
        assert result.status != AnalysisStatus.LLM_ERROR

    # ── stream ─────────────────────────────────────────────────
    def test_stream_yields_tokens(self):
        tokens = list(self.chain.stream(SAMPLE_CONTRACT_KO))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_stream_full_text_nonempty(self):
        full = "".join(self.chain.stream(SAMPLE_CONTRACT_KO))
        assert len(full) > 100

    # ── ainvoke ────────────────────────────────────────────────
    def test_ainvoke_korean(self):
        result = asyncio.run(self.chain.ainvoke(SAMPLE_CONTRACT_KO))
        assert result.ok

    def test_ainvoke_returns_analysis_result(self):
        result = asyncio.run(self.chain.ainvoke(SAMPLE_CONTRACT_EN))
        assert isinstance(result, AnalysisResult)

    # ── batch ──────────────────────────────────────────────────
    def test_batch_returns_correct_count(self):
        texts = [SAMPLE_CONTRACT_KO, SAMPLE_CONTRACT_EN]
        results = self.chain.batch(texts, max_concurrency=2)
        assert len(results) == 2

    def test_batch_all_have_status(self):
        results = self.chain.batch([SAMPLE_CONTRACT_KO, SAMPLE_CONTRACT_EN])
        for r in results:
            assert isinstance(r.status, AnalysisStatus)

    def test_batch_debug_info_has_index(self):
        results = self.chain.batch([SAMPLE_CONTRACT_KO, SAMPLE_CONTRACT_EN])
        for i, r in enumerate(results):
            if r.debug_info:
                assert r.debug_info.get("batch_index") == i

    # ── 파싱 실패 vs 분석 없음 구분 ───────────────────────────────
    def test_status_distinction(self):
        """파싱 실패와 분석 결과 없음이 명확히 구분되어야 한다."""
        result = self.chain.invoke(SAMPLE_CONTRACT_KO)
        if result.status == AnalysisStatus.PARSE_ERROR:
            assert result.parse_error != ""
            assert result.output is None
        elif result.status == AnalysisStatus.NO_FINDINGS:
            assert result.output is not None
            assert result.findings == []
        elif result.status == AnalysisStatus.SUCCESS:
            assert result.output is not None
            assert len(result.findings) > 0
