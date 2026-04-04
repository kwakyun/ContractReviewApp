import json
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ValidationError

from src.models import ContractAnalysisOutput, RiskFinding
from src.tools import ALL_TOOLS, TOOLS_BY_NAME

load_dotenv()

LLM_MODEL = "gpt-5-mini"

ANALYSIS_SYSTEM = """당신은 계약서 법무 검토 전문 AI입니다.
계약서 원문을 분석하여 위험 조항을 탐지하고, 반드시 지정된 JSON 스키마로만 응답하세요.

규칙:
- 모든 리스크 판단에는 반드시 원문 근거(evidence)를 포함하세요.
- 법률 자문을 단정하는 표현은 사용하지 마세요.
- 한국어·영어 계약서 모두 처리 가능합니다.
- confidence가 낮으면 0.5 이하로 표시하세요."""


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_completion_tokens=4096,
    )


# ── 분석 전용 LLM: with_structured_output ─────────────────────────
def get_analysis_llm():
    llm = get_llm()
    return llm.with_structured_output(ContractAnalysisOutput, method="function_calling")


# ── Tool-calling 전용 에이전트 ─────────────────────────────────────
def build_tool_agent():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def llm_call(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def tool_node(state: MessagesState):
        results = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_fn = TOOLS_BY_NAME.get(tool_call["name"])
            try:
                observation = tool_fn.invoke(tool_call["args"]) if tool_fn else f"알 수 없는 도구: {tool_call['name']}"
            except Exception as e:
                observation = f"도구 실행 오류: {str(e)}"
            results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        return {"messages": results}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tool_node" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    graph = StateGraph(MessagesState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph.add_edge("tool_node", "llm_call")
    return graph.compile()


_tool_agent = None


def get_tool_agent():
    global _tool_agent
    if _tool_agent is None:
        _tool_agent = build_tool_agent()
    return _tool_agent


# ── 계약서 분석 (structured output) ──────────────────────────────
def analyze_contract(contract_text: str) -> ContractAnalysisOutput:
    """
    계약서를 분석하여 ContractAnalysisOutput 반환.
    1차: with_structured_output (보장된 스키마)
    2차 fallback: 직접 JSON 파싱
    """
    truncated = contract_text[:12000]

    # 1차 시도: with_structured_output
    try:
        analysis_llm = get_analysis_llm()
        result = analysis_llm.invoke([
            SystemMessage(content=ANALYSIS_SYSTEM),
            HumanMessage(content=f"다음 계약서를 분석하세요:\n\n---\n{truncated}\n---"),
        ])
        if isinstance(result, ContractAnalysisOutput):
            return result
        if isinstance(result, dict):
            return ContractAnalysisOutput(**result)
    except Exception as e1:
        print(f"[structured_output 실패] {e1}")

    # 2차 fallback: JSON 프롬프트 직접 파싱
    try:
        llm = get_llm()
        schema = ContractAnalysisOutput.model_json_schema()
        prompt = f"""다음 계약서를 분석하고, 아래 JSON 스키마에 맞는 JSON 객체만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.

스키마:
{json.dumps(schema, ensure_ascii=False, indent=2)}

계약서:
---
{truncated}
---

JSON:"""
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)

        # 코드블록 제거
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()

        # JSON 객체 추출 (최외곽 { } 탐색)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            return ContractAnalysisOutput(**parsed)
    except Exception as e2:
        print(f"[fallback JSON 실패] {e2}\n원본 응답: {raw if 'raw' in dir() else 'N/A'}")

    # 최종 실패 응답
    return ContractAnalysisOutput(
        contract_type="Other",
        parties={},
        findings=[
            RiskFinding(
                clause_category="Other",
                risk_level="Medium",
                issue_summary="자동 분석에 실패했습니다. 수동 검토가 필요합니다.",
                evidence="(파싱 실패 — 콘솔 로그를 확인하세요)",
                why_it_matters="LLM 응답을 구조화하지 못했습니다.",
                questions_to_ask=["법무 담당자에게 직접 검토를 요청하세요."],
                recommended_action="확인",
                confidence=0.0,
            )
        ],
    )


# ── Tool 기능들 (RAG / 정책 / 외부 검색) ─────────────────────────
def run_rag_for_finding(finding: RiskFinding) -> str:
    agent = get_tool_agent()
    query = f"{finding.clause_category.value} 관련 조항: {finding.issue_summary}\n근거: {finding.evidence}"
    result = agent.invoke({"messages": [
        SystemMessage(content="search_standard_clauses 도구를 사용하여 유사한 표준 조항을 검색하고, 현재 문구와 비교하여 추천 수정안을 제시하세요."),
        HumanMessage(content=f"다음 리스크 조항에 대한 표준 조항을 검색하고 수정안을 제시하세요:\n\n{query}"),
    ]})
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


def run_policy_check(finding: RiskFinding) -> str:
    agent = get_tool_agent()
    result = agent.invoke({"messages": [
        SystemMessage(content="check_company_policy 도구를 사용하여 사내 정책을 조회하고 충돌 여부를 분석하세요."),
        HumanMessage(content=f"카테고리 '{finding.clause_category.value}'에 대한 사내 정책을 조회하고 아래 조항과 충돌 여부를 분석하세요:\n\n{finding.evidence}"),
    ]})
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


def run_external_search(query: str) -> str:
    agent = get_tool_agent()
    result = agent.invoke({"messages": [
        SystemMessage(content="search_external_regulations 도구를 사용하여 최신 법령/규정을 검색하고 출처와 함께 요약하세요."),
        HumanMessage(content=f"다음 주제에 대해 최신 규정을 검색하세요: {query}"),
    ]})
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)
