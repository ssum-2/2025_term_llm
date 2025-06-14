import asyncio
import os, re
import pickle
from typing import TypedDict, List, Dict, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from vector_db import save_to_vector_db, search_by_metadata
from utils import parse_pdf, summarize_single_element
from chains import (
    build_extractor_chain,
    build_report_summary_chain, # 새로 추가
    build_news_summary_chain, # 새로 추가
    ReportSummary, # 새로 추가
    NewsSummary, # 새로 추가
)
from financial import get_financial_timeseries_data_for_llm, FinancialTimeSeriesData
from news_crawler import get_latest_news
from aiolimiter import AsyncLimiter

limiter = AsyncLimiter(max_rate=50, time_period=60)

# --- 1. State 확장: 모든 분석 데이터를 담는 중앙 저장소 역할 ---
class AnalysisState(TypedDict):
    pdf_paths: List[str]
    pdf_hashes: List[str]  # [수정] pdf_hash -> pdf_hashes (복수형)
    mode: str
    semaphore: asyncio.Semaphore
    llm: BaseChatModel
    mini_llm: BaseChatModel

    # 데이터 처리 결과
    all_docs: List[Document]
    detected_stocks: Dict[str, str]
    financial_data: Dict[str, FinancialTimeSeriesData]
    news_articles: Dict[str, List[dict]]

    # 초기 분석 결과물
    report_summary: Optional[ReportSummary]
    news_summary: Optional[NewsSummary]


async def process_single_pdf(pdf_path, pdf_hash, state: AnalysisState):
    """개별 PDF에 대한 파싱, 요약, 추출을 수행하는 헬퍼 함수"""
    mode = state['mode']
    semaphore = state['semaphore']
    llm = state['llm']
    mini_llm = state['mini_llm']

    CACHE_DIR_PARSE = "./.cache/parsed_pdfs"
    cache_file_path_parse = os.path.join(CACHE_DIR_PARSE, f"{pdf_hash}.pkl")
    if os.path.exists(cache_file_path_parse):
        with open(cache_file_path_parse, "rb") as f:
            elements = pickle.load(f)
    else:
        elements = parse_pdf(pdf_path)
        os.makedirs(CACHE_DIR_PARSE, exist_ok=True)
        with open(cache_file_path_parse, "wb") as f:
            pickle.dump(elements, f)

    CACHE_DIR_SUMMARIZE = "./.cache/summarized_docs"
    cache_file_path_summarize = os.path.join(CACHE_DIR_SUMMARIZE, f"{pdf_hash}.pkl")
    if os.path.exists(cache_file_path_summarize):
        with open(cache_file_path_summarize, "rb") as f:
            processed_docs = pickle.load(f)
    else:
        source_file = os.path.basename(pdf_path)
        extractor_chain = build_extractor_chain(llm)

        async def process_element(element):
            async with semaphore, limiter:
                elem_type = element.get("type")
                element_info = {"pdf_path": pdf_path, "llm": mini_llm, **element}
                summarized_doc_task = asyncio.to_thread(summarize_single_element, element_info)
                extracted_docs = []
                if elem_type == "text":
                    try:
                        extracted_data = await extractor_chain.ainvoke({"text": element["content"]})
                        for card in extracted_data.cards:
                            card.source_file = source_file
                            card.source_page = str(element["page"])
                            page_content = f"증권사: {card.name}\n의견: {card.opinion}\n목표주가: {card.target_price}원\n근거: {card.rationale}"
                            metadata = {"source_file": card.source_file, "page": card.source_page, "element_type": "broker_info", "broker": card.name}
                            extracted_docs.append(Document(page_content=page_content, metadata=metadata))
                    except Exception:
                        pass
                summarized_doc = await summarized_doc_task
                return [summarized_doc] + extracted_docs if summarized_doc else extracted_docs

        tasks = [process_element(elem) for elem in elements]
        results = await asyncio.gather(*tasks)
        processed_docs = [doc for res in results if res for doc in res]
        os.makedirs(CACHE_DIR_SUMMARIZE, exist_ok=True)
        with open(cache_file_path_summarize, "wb") as f:
            pickle.dump(processed_docs, f)

    return processed_docs


async def process_pdfs_in_parallel_node(state: AnalysisState) -> dict:
    """여러 PDF 파일을 병렬 처리. state에서 'pdf_hashes'를 정확히 참조."""
    print("[Node: process_pdfs] PDF 병렬 처리 시작...")
    tasks = [
        process_single_pdf(path, h, state)
        for path, h in zip(state['pdf_paths'], state['pdf_hashes'])  # [수정] state['pdf_hash'] -> state['pdf_hashes']
    ]
    results = await asyncio.gather(*tasks)
    all_docs = [doc for res in results for doc in res]
    print(f"[Node: process_pdfs] 모든 PDF 처리 완료. 총 {len(all_docs)}개 문서 생성.")
    return {"all_docs": all_docs}


async def embedding_node(state: AnalysisState) -> dict:
    print("[Node: embedding] 벡터 DB 저장 시작...")
    await save_to_vector_db(state["all_docs"])
    return {}


def detect_main_stocks_node(state: AnalysisState) -> dict:
    print("[Node: detect_stocks] 핵심 종목 탐지 시작...")
    from main import ticker_cache
    stock_counts = {}
    full_text = " ".join(d.page_content for d in state['all_docs'])
    sorted_stock_names = sorted(ticker_cache.keys(), key=len, reverse=True)
    for stock_name in sorted_stock_names:
        count = full_text.count(stock_name)
        if count > 0:
            stock_counts[stock_name] = stock_counts.get(stock_name, 0) + count
    if not stock_counts:
        return {"detected_stocks": {}}
    top_stocks = sorted(stock_counts.items(), key=lambda item: item[1], reverse=True)[:3]
    detected_stocks = {name: ticker_cache.get(name) for name, count in top_stocks if name in ticker_cache}
    print(f"[Node: detect_stocks] 탐지된 핵심 종목: {list(detected_stocks.keys())}")
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
    """[수정됨] PDF와 뉴스 정보를 요약하기 전에, Python 코드로 데이터를 정제하고 중복을 제거합니다."""
    print("[Node: generate_initial_summaries] 초기 요약 및 데이터 정제 시작...")
    mini_llm = state['mini_llm']
    from main import ticker_cache  # 종목명 필터링을 위해 ticker_cache 임포트

    # --- 1. PDF 리포트 정보 정제 및 중복 제거 ---
    broker_docs = search_by_metadata(filter_criteria={"element_type": "broker_info"}, k=100)

    # 영어/한글 증권사명 통일 및 데이터 정제를 위한 로직
    normalized_brokers: Dict[str, Dict] = {}

    # 증권사 이름 통일 규칙 (필요에 따라 추가)
    name_map = {
        "Daishin Securities": "대신증권",
        "Hana Securities": "하나증권",
        "BNK Securities": "BNK투자증권"
        # ... 기타 영문/약칭 이름 추가
    }

    # 종목명 리스트
    stock_names = set(ticker_cache.keys())

    for doc in broker_docs:
        # doc.page_content = "증권사: {name}\n의견: {opinion}\n목표주가: {price}원\n근거: {rationale}"
        try:
            # page_content에서 구조화된 정보 다시 파싱
            lines = doc.page_content.strip().split('\n')
            broker_name = lines[0].split(': ')[1].strip()
            opinion = lines[1].split(': ')[1].strip()
            target_price = int(re.sub(r'[^0-9]', '', lines[2].split(': ')[1]))
            rationale = lines[3].split(': ')[1].strip()

            # [문제 해결 로직 1] 증권사 이름이 종목명인 경우 필터링
            if broker_name in stock_names:
                print(f"[Data Cleansing] 증권사명으로 잘못 추출된 종목명 필터링: {broker_name}")
                continue

            # [문제 해결 로직 2] 증권사 이름 통일 (한/영문)
            normalized_name = name_map.get(broker_name, broker_name)

            # [문제 해결 로직 3] 정보 종합 및 중복 제거
            if normalized_name not in normalized_brokers:
                normalized_brokers[normalized_name] = {
                    "opinions": [],
                    "prices": [],
                    "rationales": [],
                    "sources": []
                }

            normalized_brokers[normalized_name]["opinions"].append(opinion)
            normalized_brokers[normalized_name]["prices"].append(target_price)
            normalized_brokers[normalized_name]["rationales"].append(rationale)
            normalized_brokers[normalized_name]["sources"].append(
                f"{doc.metadata.get('source_file')}, p.{doc.metadata.get('page')}")

        except (IndexError, ValueError) as e:
            # doc.page_content 형식이 예상과 다를 경우 건너뛰기
            print(f"[Data Cleansing] 브로커 정보 파싱 실패 (내용: {doc.page_content.strip()}), 에러: {e}")
            continue

    # 정제된 데이터를 ReportSummary의 companies 형식으로 변환
    from chains import Company  # Pydantic 모델 임포트
    final_companies = []
    for name, data in normalized_brokers.items():
        # 가장 빈번한 의견, 가장 높은 목표가 등을 선택하는 로직
        from collections import Counter
        final_opinion = Counter(data['opinions']).most_common(1)[0][0] if data['opinions'] else "N/A"
        final_price = max(data['prices']) if data['prices'] else 0

        final_companies.append(Company(
            name=name,
            opinion=final_opinion,
            target_price=final_price,
            rationale=". ".join(list(set(data['rationales']))),  # 중복 근거 제거
            source_file=list(set(s.split(',')[0] for s in data['sources']))[0],  # 대표 소스파일 하나
            source_page=", ".join(list(set(s.split('p.')[1] for s in data['sources'] if 'p.' in s)))  # 페이지 번호들
        ))

    # LLM에게는 이렇게 정제된 데이터를 전달하여 요약만 요청
    report_summary_chain = build_report_summary_chain(mini_llm)
    # Pydantic 객체를 LLM 친화적인 문자열로 변환
    context_for_report = "\n\n".join([c.model_dump_json(indent=2) for c in final_companies])
    report_summary_task = report_summary_chain.ainvoke({"context": context_for_report})

    # --- 2. 뉴스 요약 (기존과 동일) ---
    all_news = [
        article for articles_list in state.get('news_articles', {}).values() for article in articles_list
    ]
    news_summary_chain = build_news_summary_chain(mini_llm)
    news_summary_task = news_summary_chain.ainvoke({"news_articles": all_news})

    # 병렬 실행
    report_summary_result, news_summary_result = await asyncio.gather(report_summary_task, news_summary_task)

    # 최종 결과물에 파이썬으로 정제한 companies 리스트를 직접 할당
    if report_summary_result:
        report_summary_result.companies = final_companies

    print("[Node: generate_initial_summaries] 초기 요약 및 데이터 정제 완료.")
    return {"report_summary": report_summary_result, "news_summary": news_summary_result}


async def run_analysis_workflow(
    pdf_paths: List[str],
    pdf_hashes: List[str],
    mode: str,
    semaphore: asyncio.Semaphore,
    llm: BaseChatModel,
    mini_llm: BaseChatModel,
):
    """모든 분석 단계를 포함하는 LangGraph 워크플로우를 실행합니다."""
    workflow = StateGraph(AnalysisState)

    # 노드 추가
    workflow.add_node("process_pdfs", process_pdfs_in_parallel_node)
    workflow.add_node("embed", embedding_node)
    workflow.add_node("detect_stocks", detect_main_stocks_node)
    workflow.add_node("fetch_external_data", fetch_external_data_node)
    workflow.add_node("generate_initial_summaries", generate_initial_summaries_node)

    # 엣지 연결
    workflow.add_edge(START, "process_pdfs")
    workflow.add_edge("process_pdfs", "embed")
    workflow.add_edge("embed", "detect_stocks")
    workflow.add_edge("detect_stocks", "fetch_external_data")
    workflow.add_edge("fetch_external_data", "generate_initial_summaries")
    workflow.add_edge("generate_initial_summaries", END)

    app = workflow.compile()

    # 워크플로우의 첫 상태를 정의하는 부분
    initial_state = {
        "pdf_paths": pdf_paths,
        "pdf_hashes": pdf_hashes, # [수정] 변수명 통일
        "mode": mode,
        "semaphore": semaphore,
        "llm": llm,
        "mini_llm": mini_llm,
    }

    final_state = await app.ainvoke(initial_state)
    print(f"--- 워크플로우 완료: {len(pdf_paths)}개 PDF 처리 완료 ---")
    return final_state