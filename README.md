# 계약서 검토 보조 · ContractReviewApp

계약서에서 검토할 조항을 찾고 원문 근거와 함께 구조화된 결과로 표현하는 AI 학습 프로젝트입니다.
Streamlit 화면에서 문서를 입력하고 분석 결과, 표준 조항 비교, JSON·Markdown 보고서를 살펴볼 수 있습니다.

## 주요 기능과 코드

| 기능 | 구현 위치 |
| --- | --- |
| 문서 입력과 분석 화면 | [app.py](app.py), [문서 파서](src/document_parser.py) |
| 구조화 분석과 결과 상태 | [분석 체인](contracts/chain.py), [스키마](contracts/schema.py) |
| 도구 호출과 분석 보조 | [agent.py](src/agent.py), [tools.py](src/tools.py) |
| 표준 조항 검색과 보고서 | [vectorstore.py](src/vectorstore.py), [report.py](src/report.py) |
| 스키마·실패 상태·분석 체인 테스트 | [test_chain.py](tests/test_chain.py) |

기술: Python, Streamlit, LangChain, LangGraph, Pydantic, ChromaDB.

## 실행 방법

Python 가상환경에서 실행합니다. 아래는 PowerShell 기준입니다.

~~~powershell
git clone https://github.com/kwakyun/ContractReviewApp.git
cd ContractReviewApp
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python -m streamlit run app.py
~~~

로컬 .env에 OPENAI_API_KEY를 설정해야 실제 분석이 가능합니다. 외부 검색에는 TAVILY_API_KEY가 추가로 필요합니다.
화면에서 합성한 짧은 TXT 계약서를 입력해 분석 흐름을 확인할 수 있습니다.
테스트에 포함된 한국어·영어 예시 문장은 [테스트 파일](tests/test_chain.py)을 참고하세요.

## 검증과 제한 사항

~~~powershell
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest tests -v -m "not live"
~~~

- 실제 LLM 호출 테스트에는 live 마커가 있습니다. 위 명령은 해당 테스트를 제외합니다.
- 의존성 전체 설치와 외부 API 연동은 이번 문서 정리에서 실행 검증하지 않았습니다. requirements.txt의 버전 호환성도 확인이 필요합니다.
- AI 결과는 법률 자문이 아닙니다. 결과와 원문을 함께 검토해야 합니다.
- 민감한 실제 계약서를 입력하기 전에 외부 AI 전송 범위를 확인해야 합니다.

## 더 읽기

[원래 기획](raw_spec.md) · [AI 활용 기록](AI_NOTES.md) · [변경 기록](CHANGELOG.md) · [작업 방법](CONTRIBUTING.md)
