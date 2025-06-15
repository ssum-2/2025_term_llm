import asyncio, os, re, pickle
from typing import TypedDict, List, Dict, Optional
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from vector_db import save_to_vector_db, search_by_metadata
from utils import parse_pdf, summarize_single_element
from chains import (
    build_extractor_chain,
    build_report_summary_chain,
    build_news_summary_chain,
    ReportSummary,
    NewsSummary,
    KeyAssertion,
)
from financial import get_financial_timeseries_data_for_llm, FinancialTimeSeriesData
from news_crawler import get_latest_news
from aiolimiter import AsyncLimiter

limiter = AsyncLimiter(max_rate=50, time_period=60)

# --- 1. State 확장: 모든 분석 데이터를 담는 중앙 저장소 역할 ---
class AnalysisState(TypedDict):
    pdf_paths: List[str]
    pdf_hashes: List[str]
    mode: str
    semaphore: asyncio.Semaphore
    llm: BaseChatModel
    mini_llm: BaseChatModel
    all_docs: List[Document]
    detected_stocks: Dict[str, str]
    financial_data: Dict[str, FinancialTimeSeriesData]
    news_articles: Dict[str, List[dict]]
    report_summary: Optional[ReportSummary]
    news_summary: Optional[NewsSummary]


async def process_single_pdf(pdf_path, pdf_hash, state: AnalysisState):
    mode = state['mode']
    semaphore = state['semaphore']
    llm = state['llm']
    mini_llm = state['mini_llm']

    CACHE_DIR_SUMMARIZE = "./.cache/summarized_docs"
    cache_file_path = os.path.join(CACHE_DIR_SUMMARIZE, f"{pdf_hash}.pkl")

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "rb") as f:
            return pickle.load(f)

    elements = parse_pdf(pdf_path)
    source_file = os.path.basename(pdf_path)
    extractor_chain = build_extractor_chain(llm)

    async def process_element(element):
        async with semaphore, limiter:
            element_info = {"pdf_path": pdf_path, "llm": mini_llm, **element}
            summarized_doc_task = asyncio.to_thread(summarize_single_element, element_info)
            extracted_docs = []
            if element.get("type") == "text":
                try:
                    extracted_data = await extractor_chain.ainvoke({"text": element["content"]})
                    for assertion in extracted_data.assertions:
                        assertion.source_file = source_file
                        assertion.source_page = str(element["page"]).replace("chunk_", "")
                        page_content = f"주장: {assertion.assertion}\n근거: {assertion.evidence}"
                        metadata = {
                            "source_file": assertion.source_file, "page": assertion.source_page,
                            "element_type": "assertion_info", "source_entity": assertion.source_entity,
                            "subject": assertion.subject, "sentiment": assertion.sentiment,
                        }
                        extracted_docs.append(Document(page_content=page_content, metadata=metadata))
                except Exception as e:
                    print(f"Extractor_chain 실패: {e}")
            summarized_doc = await summarized_doc_task
            return [summarized_doc] + extracted_docs if summarized_doc else extracted_docs

    tasks = [process_element(elem) for elem in elements]
    results = await asyncio.gather(*tasks)
    processed_docs = [doc for res in results if res for doc in res]

    os.makedirs(CACHE_DIR_SUMMARIZE, exist_ok=True)
    with open(cache_file_path, "wb") as f:
        pickle.dump(processed_docs, f)
    return processed_docs

async def process_pdfs_in_parallel_node(state: AnalysisState) -> dict:
    print("[Node: process_pdfs] PDF 병렬 처리 시작...")
    tasks = [process_single_pdf(path, h, state) for path, h in zip(state['pdf_paths'], state['pdf_hashes'])]
    results = await asyncio.gather(*tasks)
    all_docs = [doc for res in results for doc in res]
    print(f"[Node: process_pdfs] 모든 PDF 처리 완료. 총 {len(all_docs)}개 문서 생성.")
    return {"all_docs": all_docs}

async def embedding_node(state: AnalysisState) -> dict:
    print("[Node: embedding] 벡터 DB 저장 시작...")
    await save_to_vector_db(state["all_docs"])
    return {}


def detect_main_stocks_node(state: AnalysisState) -> dict:
    """'핵심 주장'의 주제(subject)를 기반으로 종목 탐지 정확도 대폭 개선"""
    print("[Node: detect_stocks] 개선된 핵심 종목 탐지 시작...")
    from main import ticker_cache

    stock_counts = defaultdict(int)

    # 1. 벡터 DB에서 모든 '핵심 주장(assertion)' 정보를 가져옵니다.
    assertion_docs = search_by_metadata(filter_criteria={"element_type": "assertion_info"}, k=500)
    if not assertion_docs:
        print("[Node: detect_stocks] 분석된 '핵심 주장' 정보가 없어 종목 탐지를 중단합니다.")
        return {"detected_stocks": {}}

    print(f"[Node: detect_stocks] {len(assertion_docs)}개의 핵심 주장을 기반으로 분석 중...")

    # 2. 각 주장의 '주제(subject)' 메타데이터에서 종목명을 카운트합니다.
    #    (예: "SK하이닉스 HBM 사업 전망" -> SK하이닉스 카운트 +1)
    for doc in assertion_docs:
        subject = doc.metadata.get('subject', '')
        for stock_name in ticker_cache.keys():
            if stock_name in subject:
                stock_counts[stock_name] += 1

    # 3. 파일명에서도 종목명을 탐지하여 가중치를 부여합니다.
    for path in state.get('pdf_paths', []):
        filename = os.path.basename(path)
        for stock_name in ticker_cache.keys():
            if stock_name in filename:
                print(f"[Detect Stocks] 파일명 '{filename}'에서 '{stock_name}' 발견. 가중치 +10 부여.")
                stock_counts[stock_name] += 10

    if not stock_counts:
        print("[Node: detect_stocks] 유효한 종목을 찾지 못했습니다.")
        return {"detected_stocks": {}}

    # 4. 가장 많이 언급된 상위 3개 종목을 최종 선택합니다.
    top_stocks = sorted(stock_counts.items(), key=lambda item: item[1], reverse=True)[:3]

    detected_stocks = {name: ticker_cache.get(name) for name, count in top_stocks if name in ticker_cache}

    print(f"[Node: detect_stocks] 탐지된 핵심 종목 (점수 포함): {top_stocks}")
    print(f"[Node: detect_stocks] 최종 선택된 종목: {list(detected_stocks.keys())}")

    return {"detected_stocks": detected_stocks}

async def fetch_external_data_node(state: AnalysisState) -> dict:
    print("[Node: fetch_external_data] 재무 데이터 및 뉴스 수집 시작...")
    detected_stocks = state.get('detected_stocks', {})
    if not detected_stocks:
        return {"financial_data": {}, "news_articles": {}}
    financial_tasks = {ticker: asyncio.to_thread(get_financial_timeseries_data_for_llm, ticker) for ticker in detected_stocks.values() if ticker}
    news_tasks = {stock_name: asyncio.to_thread(get_latest_news, stock_name, max_results=5) for stock_name in detected_stocks.keys()}
    financial_results = {ticker: await task for ticker, task in financial_tasks.items()}
    news_results = {name: await task for name, task in news_tasks.items()}
    print("[Node: fetch_external_data] 데이터 수집 완료.")
    return {"financial_data": financial_results, "news_articles": news_results}


async def generate_initial_summaries_node(state: AnalysisState) -> dict:
    print("[Node: generate_initial_summaries] 주제별 주장 그룹화 및 요약 시작...")
    mini_llm = state['mini_llm']

    # 1. PDF에서 추출된 모든 '핵심 주장(Assertion)' 정보 취합 및 그룹화
    assertion_docs = search_by_metadata(filter_criteria={"element_type": "assertion_info"}, k=500)
    assertions_by_theme = defaultdict(list)
    for doc in assertion_docs:
        try:
            assertion = KeyAssertion(
                subject=doc.metadata['subject'],
                assertion=doc.page_content.split('\n')[0].replace("주장: ", "").strip(),
                evidence=doc.page_content.split('\n')[1].replace("근거: ", "").strip(),
                sentiment=doc.metadata['sentiment'],
                source_entity=doc.metadata['source_entity'],
                source_file=doc.metadata['source_file'],
                source_page=doc.metadata['page'],
            )
            assertions_by_theme[assertion.subject].append(assertion)
        except Exception as e:
            print(f"[Data Grouping] Assertion 정보 복원 실패: {doc.metadata}, 에러: {e}")
            continue

    # 2. LLM에게 종합 인사이트 생성 요청
    context_for_llm = ""
    for theme, assertions in assertions_by_theme.items():
        context_for_llm += f"### 주제: {theme}\n"
        for ast in assertions:
            context_for_llm += f"- [{ast.sentiment}/{ast.source_entity}] {ast.assertion} (근거: {ast.evidence})\n"
        context_for_llm += "\n"

    report_summary_chain = build_report_summary_chain(mini_llm)
    report_summary_task = report_summary_chain.ainvoke({"context": context_for_llm})

    # 3. 뉴스 요약
    all_news = [article for articles_list in state.get('news_articles', {}).values() for article in articles_list]
    news_summary_chain = build_news_summary_chain(mini_llm)
    news_summary_task = news_summary_chain.ainvoke({"news_articles": all_news})

    # 병렬 실행
    report_summary_result, news_summary_result = await asyncio.gather(report_summary_task, news_summary_task)

    # 4. LLM 결과와 Python 그룹화 결과를 합쳐 최종 ReportSummary 객체 완성
    if report_summary_result:
        report_summary_result.assertions_by_theme = dict(assertions_by_theme)
    else:  # LLM이 실패한 경우 대비
        report_summary_result = ReportSummary(overall_insight="종합 인사이트 생성에 실패했습니다.",
                                              assertions_by_theme=dict(assertions_by_theme))

    print("[Node: generate_initial_summaries] 주제별 요약 완료.")
    return {"report_summary": report_summary_result, "news_summary": news_summary_result}


async def run_analysis_workflow(pdf_paths: List[str], pdf_hashes: List[str], mode: str, semaphore: asyncio.Semaphore, llm: BaseChatModel, mini_llm: BaseChatModel):
    workflow = StateGraph(AnalysisState)
    workflow.add_node("process_pdfs", process_pdfs_in_parallel_node)
    workflow.add_node("embed", embedding_node)
    workflow.add_node("detect_stocks", detect_main_stocks_node)
    workflow.add_node("fetch_external_data", fetch_external_data_node)
    workflow.add_node("generate_initial_summaries", generate_initial_summaries_node)

    workflow.add_edge(START, "process_pdfs")
    workflow.add_edge("process_pdfs", "embed")
    workflow.add_edge("embed", "detect_stocks")
    workflow.add_edge("detect_stocks", "fetch_external_data")
    workflow.add_edge("fetch_external_data", "generate_initial_summaries")
    workflow.add_edge("generate_initial_summaries", END)

    app = workflow.compile()
    initial_state = {"pdf_paths": pdf_paths, "pdf_hashes": pdf_hashes, "mode": mode, "semaphore": semaphore, "llm": llm,
                     "mini_llm": mini_llm}
    final_state = await app.ainvoke(initial_state)
    print(f"--- 워크플로우 완료: {len(pdf_paths)}개 PDF 처리 완료 ---")
    return final_state