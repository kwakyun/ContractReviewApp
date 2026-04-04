import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from contracts.chain import get_chain
from contracts.schema import AnalysisResult, AnalysisStatus, RiskLevel, StandardClause
from src.agent import run_external_search, run_policy_check, run_rag_for_finding
from src.document_parser import parse_document
from src.models import (
    AnalysisReport, ContractType, ExternalSearchResult,
    PolicyCheckResult, RAGResult,
)
from src.report import generate_json_export, generate_markdown_report
from src.vectorstore import (
    add_clause_to_vectorstore, load_standard_clauses,
    search_similar_clauses, seed_vectorstore,
)

# ── 페이지 설정 ────────────────────────────────────────────────────
st.set_page_config(
    page_title="계약서 법무 검토 보조",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLOR = {RiskLevel.High: "#ff4b4b", RiskLevel.Medium: "#ffa500", RiskLevel.Low: "#21c354"}
RISK_EMOJI = {RiskLevel.High: "🔴", RiskLevel.Medium: "🟡", RiskLevel.Low: "🟢"}
STATUS_ICON = {
    AnalysisStatus.SUCCESS: "✅",
    AnalysisStatus.NO_FINDINGS: "ℹ️",
    AnalysisStatus.PARSE_ERROR: "⚠️",
    AnalysisStatus.LLM_ERROR: "❌",
}

# ── 사이드바 ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ 계약서 검토 보조")
    st.caption("법률 자문이 아닌 참고용 보조 도구입니다.")
    st.divider()

    api_ok = bool(os.getenv("OPENAI_API_KEY"))
    st.success("API 키 연결됨") if api_ok else st.warning("OPENAI_API_KEY 미설정")
    if not os.getenv("TAVILY_API_KEY"):
        st.info("TAVILY_API_KEY 미설정 — 외부 검색 비활성")

    debug_mode = st.checkbox("🐛 디버그 모드 (RAW 출력 표시)", value=False)
    st.divider()
    st.caption("GPT-5-mini · LangChain LCEL · ChromaDB")

# ── 탭 ────────────────────────────────────────────────────────────
tab_single, tab_stream, tab_batch, tab_analysis, tab_library, tab_export = st.tabs([
    "📄 단일 분석", "🌊 스트리밍", "📦 배치 분석",
    "🔍 분석 결과", "📚 라이브러리", "📤 내보내기",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1: 단일 분석  (invoke)
# ══════════════════════════════════════════════════════════════════
with tab_single:
    st.header("📄 단일 계약서 분석")
    st.caption("`chain.invoke()` — 완료 후 구조화 결과 반환")

    uploaded = st.file_uploader("계약서 업로드 (PDF / DOCX / TXT)", type=["pdf","docx","doc","txt"], key="single_upload")

    if uploaded:
        st.success(f"선택됨: **{uploaded.name}** ({uploaded.size/1024:.1f} KB)")
        if st.button("🚀 분석 시작", type="primary", use_container_width=True, key="single_run"):
            with st.spinner("문서 파싱 중..."):
                file_bytes = uploaded.read()
                doc = parse_document(file_bytes, uploaded.name)
                if doc.get("from_cache"):
                    st.info("💾 캐시 히트 — 이전 분석 결과를 재사용합니다.")

            with st.spinner("LLM 분석 중 (gpt-5-mini)..."):
                seed_vectorstore()
                chain = get_chain(debug=debug_mode)
                result: AnalysisResult = chain.invoke(doc["raw_text"])

            # 결과 저장
            st.session_state["latest_result"] = result
            st.session_state["latest_doc"] = doc

            icon = STATUS_ICON[result.status]
            if result.status == AnalysisStatus.SUCCESS:
                st.success(f"{icon} 분석 완료 — {len(result.findings)}개 위험 조항 탐지 ({result.elapsed_sec:.1f}s)")
                st.balloons()
            elif result.status == AnalysisStatus.NO_FINDINGS:
                st.info(f"{icon} 분석 완료 — 위험 조항 없음 ({result.elapsed_sec:.1f}s)")
            else:
                st.error(f"{icon} {result.error_message}")

            if debug_mode:
                with st.expander("🐛 RAW LLM 출력"):
                    st.code(result.raw_llm_output, language="json")
                if result.parse_error:
                    with st.expander("🐛 파싱 오류 상세"):
                        st.code(result.parse_error)

            if result.ok:
                st.info("👉 **분석 결과** 탭에서 리스크 카드를 확인하세요.")

# ══════════════════════════════════════════════════════════════════
# TAB 2: 스트리밍 분석  (stream → invoke)
# ══════════════════════════════════════════════════════════════════
with tab_stream:
    st.header("🌊 스트리밍 분석")
    st.caption("`chain.stream()` — 토큰 실시간 출력 → 완료 후 파싱")

    uploaded_s = st.file_uploader("계약서 업로드", type=["pdf","docx","doc","txt"], key="stream_upload")

    if uploaded_s:
        st.success(f"선택됨: **{uploaded_s.name}**")
        if st.button("🌊 스트리밍 시작", type="primary", use_container_width=True, key="stream_run"):
            with st.spinner("문서 파싱 중..."):
                file_bytes_s = uploaded_s.read()
                doc_s = parse_document(file_bytes_s, uploaded_s.name)

            st.subheader("LLM 실시간 출력")
            stream_placeholder = st.empty()
            full_text = ""

            seed_vectorstore()
            chain_s = get_chain(debug=debug_mode)

            # 토큰 스트리밍
            with st.spinner("스트리밍 중..."):
                for token in chain_s.stream(doc_s["raw_text"]):
                    full_text += token
                    stream_placeholder.markdown(
                        f"```\n{full_text[-3000:]}\n```"  # 최근 3000자만 표시
                    )

            st.success("스트리밍 완료 — 구조화 파싱 중...")

            # 파싱 (스트리밍 결과 재활용)
            with st.spinner("파싱 중..."):
                result_s: AnalysisResult = chain_s.invoke(doc_s["raw_text"])

            st.session_state["latest_result"] = result_s
            st.session_state["latest_doc"] = doc_s

            if result_s.ok:
                st.success(f"✅ 파싱 완료 — {len(result_s.findings)}개 탐지")
                st.info("👉 **분석 결과** 탭에서 확인하세요.")
            else:
                st.error(result_s.error_message)
                if debug_mode:
                    st.code(result_s.raw_llm_output, language="text")

# ══════════════════════════════════════════════════════════════════
# TAB 3: 배치 분석  (batch)
# ══════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("📦 배치 분석")
    st.caption("`chain.batch()` — 여러 계약서 병렬 분석")

    uploaded_batch = st.file_uploader(
        "계약서 여러 개 업로드",
        type=["pdf","docx","doc","txt"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    col_b1, col_b2 = st.columns(2)
    max_conc = col_b1.slider("병렬 처리 수", 1, 5, 2)

    if uploaded_batch:
        st.write(f"**{len(uploaded_batch)}개** 파일 선택됨")
        for f in uploaded_batch:
            st.caption(f"• {f.name} ({f.size/1024:.1f} KB)")

        if col_b2.button("📦 배치 분석 시작", type="primary", use_container_width=True):
            texts, names = [], []
            progress = st.progress(0, text="파일 파싱 중...")

            for i, f in enumerate(uploaded_batch):
                doc_b = parse_document(f.read(), f.name)
                texts.append(doc_b["raw_text"])
                names.append(f.name)
                progress.progress((i + 1) / len(uploaded_batch), text=f"파싱: {f.name}")

            progress.progress(100, text="LLM 배치 분석 중...")
            seed_vectorstore()
            chain_b = get_chain(debug=debug_mode)

            with st.spinner(f"{len(texts)}개 계약서 분석 중 (max_concurrency={max_conc})..."):
                batch_results: list[AnalysisResult] = chain_b.batch(texts, max_concurrency=max_conc)

            progress.empty()
            st.session_state["batch_results"] = list(zip(names, batch_results))

            # 결과 요약 테이블
            st.subheader("배치 결과 요약")
            for name, res in zip(names, batch_results):
                icon = STATUS_ICON[res.status]
                high = sum(1 for f in res.findings if f.risk_level == RiskLevel.High)
                mid  = sum(1 for f in res.findings if f.risk_level == RiskLevel.Medium)
                low  = sum(1 for f in res.findings if f.risk_level == RiskLevel.Low)
                col1, col2, col3, col4, col5 = st.columns([3,1,1,1,2])
                col1.write(f"{icon} **{name}**")
                col2.metric("🔴", high)
                col3.metric("🟡", mid)
                col4.metric("🟢", low)
                col5.write(res.status.value)

                if debug_mode and not res.ok:
                    with st.expander(f"🐛 {name} — 오류 상세"):
                        st.code(res.error_message)
                        st.code(res.raw_llm_output[:1000])

# ══════════════════════════════════════════════════════════════════
# TAB 4: 분석 결과  (공통)
# ══════════════════════════════════════════════════════════════════
with tab_analysis:
    st.header("🔍 분석 결과")

    result: AnalysisResult | None = st.session_state.get("latest_result")
    doc_data = st.session_state.get("latest_doc", {})

    if result is None:
        st.info("먼저 **단일 분석** 또는 **스트리밍** 탭에서 분석을 실행하세요.")
    elif not result.ok:
        st.error(f"{STATUS_ICON[result.status]} {result.error_message}")
        if debug_mode:
            st.code(result.raw_llm_output)
    elif result.status == AnalysisStatus.NO_FINDINGS:
        st.success("ℹ️ 분석 결과 위험 조항이 발견되지 않았습니다.")
    else:
        output = result.output
        findings = result.findings

        # 메트릭
        col1,col2,col3,col4 = st.columns(4)
        high = sum(1 for f in findings if f.risk_level == RiskLevel.High)
        mid  = sum(1 for f in findings if f.risk_level == RiskLevel.Medium)
        low  = sum(1 for f in findings if f.risk_level == RiskLevel.Low)
        col1.metric("🔴 High", high)
        col2.metric("🟡 Medium", mid)
        col3.metric("🟢 Low", low)
        col4.metric("계약 유형", output.contract_type.value)

        if output.parties:
            with st.expander("📌 계약 당사자"):
                for role, name in output.parties.items():
                    st.write(f"**{role}**: {name}")

        st.divider()

        # 필터
        filter_lvl = st.multiselect("레벨 필터", ["High","Medium","Low"], default=["High","Medium","Low"])
        filtered = [(i,f) for i,f in enumerate(findings) if f.risk_level.value in filter_lvl]

        # AnalysisReport (RAG/정책/검색 결과 포함)
        if "analysis_report" not in st.session_state:
            st.session_state["analysis_report"] = AnalysisReport(
                document_name=doc_data.get("file_name", "unknown"),
                contract_type=output.contract_type,
                parties=output.parties,
                findings=findings,
            )
        report: AnalysisReport = st.session_state["analysis_report"]

        for idx, finding in filtered:
            emoji = RISK_EMOJI[finding.risk_level]
            label = f"{emoji} [{finding.risk_level.value}] {finding.clause_category.value} — {finding.issue_summary[:55]}{'…' if len(finding.issue_summary)>55 else ''}"
            with st.expander(label):
                col_l, col_r = st.columns([2,1])
                with col_l:
                    st.markdown(f"**요약**: {finding.issue_summary}")
                    st.markdown("**근거 (원문 인용)**:")
                    st.code(finding.evidence, language=None)
                    st.markdown(f"**왜 위험한가**: {finding.why_it_matters}")
                with col_r:
                    st.markdown(f"**권장 조치**: `{finding.recommended_action.value}`")
                    st.progress(finding.confidence, text=f"신뢰도 {finding.confidence:.0%}")
                    for q in finding.questions_to_ask:
                        st.markdown(f"- {q}")

                st.divider()
                col_rag, col_pol, col_ext = st.columns(3)

                with col_rag:
                    if st.button("📚 표준 조항 비교", key=f"rag_{idx}"):
                        with st.spinner("ChromaDB RAG 검색 중..."):
                            rag_text = run_rag_for_finding(finding)
                            similar = search_similar_clauses(
                                f"{finding.clause_category.value}: {finding.issue_summary}", top_k=3
                            )
                            report.rag_results = [r for r in report.rag_results if r.finding_index != idx]
                            report.rag_results.append(RAGResult(
                                finding_index=idx, standard_clauses=similar, recommendation=rag_text
                            ))
                            st.session_state["analysis_report"] = report

                with col_pol:
                    if st.button("🏢 사내 정책 확인", key=f"pol_{idx}"):
                        with st.spinner("정책 조회 중..."):
                            pol_text = run_policy_check(finding)
                            report.policy_checks = [p for p in report.policy_checks if p.finding_index != idx]
                            report.policy_checks.append(PolicyCheckResult(
                                finding_index=idx,
                                policy_name=f"{finding.clause_category.value} 정책",
                                has_conflict=any(k in pol_text for k in ["충돌","위반","초과"]),
                                conflict_detail=pol_text[:600],
                            ))
                            st.session_state["analysis_report"] = report

                with col_ext:
                    q_input = st.text_input("외부 규정 검색어", key=f"q_{idx}", placeholder="예: 개인정보 국외이전")
                    if st.button("🌐 검색", key=f"ext_{idx}") and q_input:
                        with st.spinner("Tavily 검색 중..."):
                            search_text = run_external_search(q_input)
                            report.external_searches.append(ExternalSearchResult(
                                query=q_input, summary=search_text
                            ))
                            st.session_state["analysis_report"] = report

                # 결과 표시
                rag_d = next((r for r in report.rag_results if r.finding_index == idx), None)
                if rag_d:
                    st.markdown("#### 📚 표준 조항 비교")
                    tabs_r = st.tabs([f"조항 {i+1}" for i in range(len(rag_d.standard_clauses))])
                    for tr, sc in zip(tabs_r, rag_d.standard_clauses):
                        with tr:
                            st.markdown(f"**[{sc.category}] {sc.title}**")
                            st.info(sc.template_text)
                            if sc.guidance:
                                st.caption(f"💡 {sc.guidance}")
                    if rag_d.recommendation:
                        st.success(rag_d.recommendation)

                pc_d = next((p for p in report.policy_checks if p.finding_index == idx), None)
                if pc_d:
                    st.markdown("#### 🏢 사내 정책")
                    if pc_d.has_conflict:
                        st.error(f"⚠️ 충돌: {pc_d.conflict_detail}")
                    else:
                        st.success(f"✅ 이상 없음\n\n{pc_d.conflict_detail}")

# ══════════════════════════════════════════════════════════════════
# TAB 5: 라이브러리
# ══════════════════════════════════════════════════════════════════
with tab_library:
    st.header("📚 표준 조항 라이브러리")
    if st.button("🔄 ChromaDB 재시드"):
        with st.spinner("재구축 중..."):
            seed_vectorstore(force=True)
            st.success("재시드 완료!")

    clauses = load_standard_clauses()
    cats = sorted(set(c.category for c in clauses))
    sel_cat = st.selectbox("카테고리 필터", ["전체"] + cats)
    shown = clauses if sel_cat == "전체" else [c for c in clauses if c.category == sel_cat]

    for c in shown:
        with st.expander(f"[{c.category}] {c.title} (v{c.version})"):
            st.info(c.template_text)
            if c.guidance:
                st.caption(f"💡 {c.guidance}")
            st.caption(f"ID: {c.id}")

    st.divider()
    st.subheader("➕ 새 표준 조항 추가")
    with st.form("add_clause"):
        nid = st.text_input("ID")
        ncat = st.selectbox("카테고리", cats + ["Other"])
        ntitle = st.text_input("제목")
        ntext = st.text_area("표준 문구")
        nguide = st.text_area("가이드 (선택)")
        if st.form_submit_button("추가") and nid and ntitle and ntext:
            add_clause_to_vectorstore(StandardClause(
                id=nid, category=ncat, title=ntitle,
                template_text=ntext, guidance=nguide,
            ))
            st.success(f"'{ntitle}' 추가 완료!")

# ══════════════════════════════════════════════════════════════════
# TAB 6: 내보내기
# ══════════════════════════════════════════════════════════════════
with tab_export:
    st.header("📤 리포트 내보내기")

    report: AnalysisReport | None = st.session_state.get("analysis_report")
    batch: list | None = st.session_state.get("batch_results")

    if report is None and batch is None:
        st.info("분석을 먼저 실행하세요.")
    else:
        if report:
            st.subheader(f"단일 리포트: {report.document_name}")
            col1, col2 = st.columns(2)
            md = generate_markdown_report(report)
            col1.download_button("⬇️ 마크다운 (.md)", md.encode(), f"{report.document_name}.md", "text/markdown", use_container_width=True)
            js = generate_json_export(report)
            col2.download_button("⬇️ JSON (.json)", js.encode(), f"{report.document_name}.json", "application/json", use_container_width=True)
            with st.expander("마크다운 미리보기"):
                st.markdown(md)

        if batch:
            st.divider()
            st.subheader("배치 결과 개별 다운로드")
            for name, res in batch:
                if res.ok and res.output:
                    rpt = AnalysisReport(
                        document_name=name,
                        contract_type=res.output.contract_type,
                        parties=res.output.parties,
                        findings=res.findings,
                    )
                    md_b = generate_markdown_report(rpt)
                    st.download_button(
                        f"⬇️ {name}", md_b.encode(), f"{name}.md",
                        "text/markdown", key=f"dl_{name}",
                    )

    st.divider()
    st.caption("⚠️ 이 리포트는 AI 보조 도구로 생성되었으며, 법률 자문이 아닙니다.")
