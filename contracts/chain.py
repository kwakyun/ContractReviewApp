"""
contracts/chain.py
──────────────────
LangChain LCEL 방식의 계약서 분석 파이프라인.

파이프라인 구조:
    prompt | llm                    → raw_chain  (스트리밍용)
    prompt | llm | fixing_parser    → chain      (구조화 출력용)

지원 인터페이스:
    .invoke()   - 단일 동기 분석
    .ainvoke()  - 단일 비동기 분석
    .stream()   - LLM 토큰 실시간 스트리밍 (파싱은 완료 후)
    .batch()    - 여러 계약서 병렬/순차 분석
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator, Iterator

from dotenv import load_dotenv
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from contracts.schema import (
    AnalysisResult,
    AnalysisStatus,
    ContractAnalysisOutput,
)

load_dotenv()

logger = logging.getLogger(__name__)

LLM_MODEL = "gpt-5-mini"
MAX_CONTRACT_CHARS = 12_000

# ──────────────────────────────────────────────────────────────────
# 프롬프트 템플릿
# ──────────────────────────────────────────────────────────────────
_SYSTEM = """\
당신은 계약서 법무 검토 전문 AI입니다.
계약서 원문을 분석하여 위험 조항을 탐지하고 아래 지정 형식으로만 응답하세요.

규칙:
- 모든 리스크 판단에는 반드시 원문 근거(evidence)를 인용하세요.
- 법률 자문을 단정하는 표현은 절대 사용하지 마세요.
- 한국어·영어 계약서 모두 처리 가능합니다.
- confidence: 근거가 명확하면 0.8↑, 추정이면 0.5 이하.

{format_instructions}"""

_HUMAN = """\
다음 계약서를 분석하세요:

---
{contract_text}
---"""


# ──────────────────────────────────────────────────────────────────
# 체인 팩토리
# ──────────────────────────────────────────────────────────────────
class ContractAnalysisChain:
    """
    LCEL 기반 계약서 분석 체인.

    Usage:
        chain = ContractAnalysisChain.build()

        # 단일 동기
        result = chain.invoke(text)

        # 단일 비동기
        result = await chain.ainvoke(text)

        # 스트리밍 (토큰 단위)
        for token in chain.stream(text):
            print(token, end="", flush=True)
        result = chain.invoke(text)   # 최종 파싱

        # 배치
        results = chain.batch([text1, text2, text3])
    """

    def __init__(self, llm: ChatOpenAI, debug: bool = False):
        self.llm = llm
        self.debug = debug
        self._build()

    # ── 내부 빌드 ──────────────────────────────────────────────
    def _build(self) -> None:
        # 1) Parser
        self.parser = PydanticOutputParser(pydantic_object=ContractAnalysisOutput)

        # 2) OutputFixingParser: 파싱 실패 시 LLM에게 수정 요청 (최대 2회)
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.parser,
            llm=self.llm,
            max_retries=2,
        )

        # 3) Prompt  ← format_instructions 포함
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])

        # 4) LCEL 체인 조합  (| 연산자)
        #    raw_chain : prompt | llm            → AIMessage  (스트리밍용)
        #    chain     : prompt | llm | parser   → ContractAnalysisOutput
        self.raw_chain = self.prompt | self.llm
        self.chain = self.raw_chain | self.fixing_parser

    # ── 공통 입력 빌더 ─────────────────────────────────────────
    def _inp(self, contract_text: str) -> dict:
        return {
            "contract_text": contract_text[:MAX_CONTRACT_CHARS],
            "format_instructions": self.parser.get_format_instructions(),
        }

    # ── 디버그 로깅 ────────────────────────────────────────────
    def _log_raw(self, raw: str) -> None:
        if self.debug:
            logger.debug("[RAW LLM OUTPUT — %d chars]\n%s", len(raw), raw[:2000])

    # ─────────────────────────────────────────────────────────────
    # invoke  (동기)
    # ─────────────────────────────────────────────────────────────
    def invoke(self, contract_text: str) -> AnalysisResult:
        """단일 계약서 동기 분석."""
        raw = ""
        t0 = time.perf_counter()
        try:
            # ① raw LLM 호출
            raw_msg = self.raw_chain.invoke(self._inp(contract_text))
            raw = raw_msg.content
            self._log_raw(raw)

            # ② OutputFixingParser 로 파싱 (실패 시 LLM 자동 수정)
            output: ContractAnalysisOutput = self.fixing_parser.invoke(raw)

            status = (
                AnalysisStatus.SUCCESS if output.findings
                else AnalysisStatus.NO_FINDINGS
            )
            return AnalysisResult(
                status=status,
                output=output,
                raw_llm_output=raw,
                elapsed_sec=time.perf_counter() - t0,
                debug_info={"model": LLM_MODEL, "chars": len(contract_text)},
            )

        except Exception as e:
            logger.error("[PARSE ERROR] %s\n[RAW]\n%s", e, raw[:1000])
            return AnalysisResult(
                status=AnalysisStatus.PARSE_ERROR,
                raw_llm_output=raw,
                parse_error=str(e),
                elapsed_sec=time.perf_counter() - t0,
            )

    # ─────────────────────────────────────────────────────────────
    # ainvoke  (비동기)
    # ─────────────────────────────────────────────────────────────
    async def ainvoke(self, contract_text: str) -> AnalysisResult:
        """단일 계약서 비동기 분석 (.ainvoke 지원)."""
        raw = ""
        t0 = time.perf_counter()
        try:
            raw_msg = await self.raw_chain.ainvoke(self._inp(contract_text))
            raw = raw_msg.content
            self._log_raw(raw)

            output: ContractAnalysisOutput = await self.fixing_parser.ainvoke(raw)

            status = (
                AnalysisStatus.SUCCESS if output.findings
                else AnalysisStatus.NO_FINDINGS
            )
            return AnalysisResult(
                status=status,
                output=output,
                raw_llm_output=raw,
                elapsed_sec=time.perf_counter() - t0,
            )

        except Exception as e:
            logger.error("[ASYNC PARSE ERROR] %s", e)
            return AnalysisResult(
                status=AnalysisStatus.PARSE_ERROR,
                raw_llm_output=raw,
                parse_error=str(e),
                elapsed_sec=time.perf_counter() - t0,
            )

    # ─────────────────────────────────────────────────────────────
    # stream  (LLM 토큰 스트리밍)
    # ─────────────────────────────────────────────────────────────
    def stream(self, contract_text: str) -> Iterator[str]:
        """
        LLM 응답 토큰을 실시간으로 yield합니다.
        구조화 파싱은 invoke()를 별도로 호출하세요.
        Streamlit: st.write_stream(chain.stream(text))
        """
        for chunk in self.raw_chain.stream(self._inp(contract_text)):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content

    async def astream(self, contract_text: str) -> AsyncIterator[str]:
        """비동기 스트리밍."""
        async for chunk in self.raw_chain.astream(self._inp(contract_text)):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content

    # ─────────────────────────────────────────────────────────────
    # batch  (다중 계약서)
    # ─────────────────────────────────────────────────────────────
    def batch(
        self,
        contract_texts: list[str],
        max_concurrency: int = 3,
    ) -> list[AnalysisResult]:
        """
        여러 계약서를 병렬 분석합니다.
        max_concurrency: LangChain 내부 병렬 처리 수
        """
        inputs = [self._inp(t) for t in contract_texts]
        results: list[AnalysisResult] = []

        try:
            # LangChain batch: raw 일괄 호출
            raw_msgs = self.raw_chain.batch(
                inputs,
                config={"max_concurrency": max_concurrency},
            )
        except Exception as e:
            logger.error("[BATCH LLM ERROR] %s", e)
            return [
                AnalysisResult(status=AnalysisStatus.LLM_ERROR, llm_error=str(e))
                for _ in contract_texts
            ]

        for i, raw_msg in enumerate(raw_msgs):
            raw = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)
            self._log_raw(raw)
            try:
                output: ContractAnalysisOutput = self.fixing_parser.invoke(raw)
                status = (
                    AnalysisStatus.SUCCESS if output.findings
                    else AnalysisStatus.NO_FINDINGS
                )
                results.append(AnalysisResult(
                    status=status,
                    output=output,
                    raw_llm_output=raw,
                    debug_info={"batch_index": i},
                ))
            except Exception as e:
                logger.error("[BATCH PARSE ERROR #%d] %s\n[RAW]\n%s", i, e, raw[:500])
                results.append(AnalysisResult(
                    status=AnalysisStatus.PARSE_ERROR,
                    raw_llm_output=raw,
                    parse_error=str(e),
                    debug_info={"batch_index": i},
                ))

        return results

    # ─────────────────────────────────────────────────────────────
    # 팩토리
    # ─────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, debug: bool = False) -> "ContractAnalysisChain":
        """기본 설정으로 체인 인스턴스 생성."""
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_completion_tokens=4096,
        )
        return cls(llm=llm, debug=debug)


# ──────────────────────────────────────────────────────────────────
# 모듈 레벨 싱글톤 (Streamlit 캐시와 호환)
# ──────────────────────────────────────────────────────────────────
_chain: ContractAnalysisChain | None = None


def get_chain(debug: bool = False) -> ContractAnalysisChain:
    global _chain
    if _chain is None:
        _chain = ContractAnalysisChain.build(debug=debug)
    return _chain
