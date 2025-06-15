import asyncio, os, re, pickle
from typing import TypedDict, List, Dict, Optional
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

from vector_db import save_to_vector_db
from utils import parse_pdf, summarize_single_element
from chains import (
    build_extractor_chain,
    build_entity_summary_chain,
    build_news_summary_chain,
    build_grand_summary_chain,
    ReportSummary, NewsSummary, ExtractedAssertion, EntityAnalysis, GrandSummary
)
from financial import get_financial_timeseries_data_for_llm, FinancialTimeSeriesData
from news_crawler import get_latest_news
from aiolimiter import AsyncLimiter

limiter = AsyncLimiter(max_rate=50, time_period=60)


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
    grand_summary: Optional[GrandSummary]

async def process_single_pdf(pdf_path, pdf_hash, state: AnalysisState):
    CACHE_DIR = "./.cache/summarized_docs"
    cache_file_path = os.path.join(CACHE_DIR, f"{pdf_hash}.pkl")
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "rb") as f: return pickle.load(f)

    elements = parse_pdf(pdf_path)
    source_file = os.path.basename(pdf_path)
    extractor_chain = build_extractor_chain(state['llm'])

    async def process_element(element):
        async with state['semaphore'], limiter:
            element_info = {"pdf_path": pdf_path, "llm": state['mini_llm'], **element}
            summarized_doc_task = asyncio.to_thread(summarize_single_element, element_info)
            extracted_docs = []
            if element.get("type") == "text":
                try:
                    extracted_data = await extractor_chain.ainvoke({"text": element["content"]})
                    for assertion in extracted_data.assertions:
                        page_content = f"주장: {assertion.assertion}\n근거: {assertion.evidence}"
                        metadata = {
                            "source_file": source_file, "page": str(element["page"]).replace("chunk_", ""),
                            "element_type": "assertion_info", "source_entity": assertion.source_entity,
                            "subject": assertion.subject, "sentiment": assertion.sentiment,
                        }
                        extracted_docs.append(Document(page_content=page_content, metadata=metadata))
                except Exception as e:
                    print(f"[Extractor Chain Error] {e}")

            summarized_doc = await summarized_doc_task
            return ([summarized_doc] if summarized_doc else []) + extracted_docs

    tasks = [process_element(elem) for elem in elements]
    results = await asyncio.gather(*tasks)
    processed_docs = [doc for res in results for doc in res if doc]
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file_path, "wb") as f:
        pickle.dump(processed_docs, f)
    return processed_docs


async def process_pdfs_node(state: AnalysisState) -> dict:
    """
    모든 PDF를 처리하는 노드. 두 가지 주요 작업을 수행합니다:
    1. 각 PDF의 개별 요소(텍스트, 테이블 등)를 요약하여 RAG 검색을 위한 기본 문서(Document)를 생성합니다.
    2. 각 PDF의 전체 텍스트에서 구조화된 '주장(Assertion)'을 추출하여 '분석 주체별 요약'을 위한 심층 문서(assertion_info)를 생성합니다.
    """
    llm = state['llm']
    mini_llm = state['mini_llm']
    semaphore = state['semaphore']
    extractor_chain = build_extractor_chain(llm)
    all_docs = []

    # --- Step 1 & 2: PDF 파싱, 개별 요소 요약 (for RAG), 전체 텍스트 추출 (for Assertion) ---
    async def parse_and_prepare(pdf_path, pdf_hash):
        CACHE_DIR = "./.cache/summarized_docs"
        cache_file_path = os.path.join(CACHE_DIR, f"{pdf_hash}.pkl")
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "rb") as f:
                return pickle.load(f)

        elements = parse_pdf(pdf_path)
        source_file = os.path.basename(pdf_path)

        # 개별 요소 요약 (RAG용)
        async def process_element(element):
            async with semaphore, limiter:
                element_info = {"pdf_path": pdf_path, "llm": mini_llm, **element}
                return await asyncio.to_thread(summarize_single_element, element_info)

        summarize_tasks = [process_element(elem) for elem in elements]
        summarized_docs = [doc for doc in await asyncio.gather(*summarize_tasks) if doc]

        # 전체 텍스트 추출 (주장 추출용)
        full_text = " ".join([elem['content'] for elem in elements if elem['type'] == 'text'])

        payload = (summarized_docs, full_text, source_file)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file_path, "wb") as f: pickle.dump(payload, f)

        return payload

    parse_tasks = [parse_and_prepare(path, h) for path, h in zip(state['pdf_paths'], state['pdf_hashes'])]
    parse_results = await asyncio.gather(*parse_tasks)

    # RAG용 요약 문서를 all_docs에 추가
    for summarized_docs, _, _ in parse_results:
        all_docs.extend(summarized_docs)

    # --- Step 3: 각 PDF의 전체 텍스트에서 '주장' 병렬 추출 ---
    async def extract_assertions_from_full_text(full_text, source_file):
        if not full_text: return []
        print(f"[Extractor] '{source_file}'의 전체 텍스트에서 주장 추출 시작...")
        try:
            extracted_data = await extractor_chain.ainvoke({"text": full_text})
            assertion_docs = []
            for assertion in extracted_data.assertions:
                page_content = f"주장: {assertion.assertion}\n근거: {assertion.evidence}"
                metadata = {
                    "source_file": source_file, "page": "N/A (전체 요약)",
                    "element_type": "assertion_info", "source_entity": assertion.source_entity,
                    "subject": assertion.subject, "sentiment": assertion.sentiment,
                }
                assertion_docs.append(Document(page_content=page_content, metadata=metadata))
            print(f"[Extractor] '{source_file}'에서 {len(assertion_docs)}개의 주장 추출 완료.")
            return assertion_docs
        except Exception as e:
            print(f"[Extractor Chain Error on {source_file}] {e}")
            return []

    extract_tasks = [extract_assertions_from_full_text(full_text, source_file) for _, full_text, source_file in
                     parse_results]
    extract_results = await asyncio.gather(*extract_tasks)

    # 추출된 주장 문서를 all_docs에 추가
    for assertion_docs in extract_results:
        all_docs.extend(assertion_docs)

    await save_to_vector_db(all_docs)
    return {"all_docs": all_docs}


async def detect_stocks_node(state: AnalysisState) -> dict:
    from main import ticker_cache
    if not ticker_cache: return {"detected_stocks": {}}

    stock_counts = defaultdict(int)
    all_docs = state['all_docs']

    # 1. 파일명에서 가중치 부여
    for path in state['pdf_paths']:
        filename = os.path.basename(path)
        for stock_name in ticker_cache.keys():
            if stock_name in filename:
                stock_counts[stock_name] += 50

    # 2. 'assertion'의 'subject' 메타데이터에서 카운트
    assertion_docs = [doc for doc in all_docs if doc.metadata.get("element_type") == "assertion_info"]
    for doc in assertion_docs:
        subject = doc.metadata.get('subject', '')
        for stock_name in ticker_cache.keys():
            if stock_name in subject:
                stock_counts[stock_name] += 5

    # 3. 일반 텍스트 요약본에서 종목명 언급 빈도 카운트
    text_docs_content = " ".join([doc.page_content for doc in all_docs if doc.metadata.get("element_type") == "text"])
    for stock_name in ticker_cache.keys():
        stock_counts[stock_name] += text_docs_content.count(stock_name)

    # 증권사, 자산운용사 등 금융기관명 제외
    valid_counts = {
        name: score for name, score in stock_counts.items()
        if "증권" not in name and "자산운용" not in name and "투자" not in name and "금융" not in name and score > 0
    }

    if not valid_counts:
        print("[Node: detect_stocks] 탐지된 종목 없음.")
        return {"detected_stocks": {}}

    top_stocks = sorted(valid_counts.items(), key=lambda item: item[1], reverse=True)[:3]
    detected_stocks = {name: ticker_cache[name] for name, count in top_stocks}
    print(f"[Node: detect_stocks] 최종 탐지된 종목: {list(detected_stocks.keys())}")
    return {"detected_stocks": detected_stocks}


async def fetch_external_data_node(state: AnalysisState) -> dict:
    detected_stocks = state.get('detected_stocks', {})
    if not detected_stocks:
        print("[Node: fetch_external_data] 탐지된 종목이 없어 외부 데이터 수집을 건너뜁니다.")
        return {"financial_data": {}, "news_articles": {}}

    financial_tasks = {name: asyncio.to_thread(get_financial_timeseries_data_for_llm, ticker) for name, ticker in detected_stocks.items()}
    news_tasks = {name: asyncio.to_thread(get_latest_news, name, max_results=5) for name in detected_stocks.keys()}

    financial_results = await asyncio.gather(*financial_tasks.values())
    news_results = await asyncio.gather(*news_tasks.values())

    financial_data = {ticker: data for data, ticker in zip(financial_results, detected_stocks.values()) if data and data.price}
    news_articles = {name: news for name, news in zip(detected_stocks.keys(), news_results) if news}

    return {"financial_data": financial_data, "news_articles": news_articles}


async def generate_summaries_node(state: AnalysisState) -> dict:
    mini_llm = state['mini_llm']
    assertion_docs = [doc for doc in state['all_docs'] if doc.metadata.get("element_type") == "assertion_info"]

    # 1. Report Summary 생성 (폴백 로직 포함)
    report_summary = None
    entity_analyses = []

    if assertion_docs:
        assertions_by_entity = defaultdict(list)
        for doc in assertion_docs:
            entity_name = doc.metadata.get('source_entity', '작성자')
            assertions_by_entity[entity_name].append(doc)

        entity_summary_chain = build_entity_summary_chain(mini_llm)
        tasks = []
        for name, docs in assertions_by_entity.items():
            if name == '작성자' and len(docs) < 2: continue
            context = "\n".join(f"- {d.page_content.split('근거:')[0].strip()}" for d in docs)
            tasks.append(entity_summary_chain.ainvoke({"entity_name": name, "claims_context": context}))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        entity_analyses = [res for res in results if isinstance(res, EntityAnalysis)]

    if entity_analyses:
        print(f"[Node: generate_summaries] 구조화된 주장 {len(entity_analyses)}개 그룹을 바탕으로 리포트 요약 생성.")
        insight_context = "\n\n".join(f"### {a.entity_name}\n- {a.main_stance}" for a in entity_analyses)
        insight_prompt = PromptTemplate.from_template(
            "다음은 여러 분석 주체들의 핵심 입장입니다. 이를 종합하여 전체 시장과 분석 대상에 대한 최종 인사이트를 3-4 문장으로 요약해주세요.\n\n{context}")
        overall_insight = (await (insight_prompt | mini_llm).ainvoke({"context": insight_context})).content
        report_summary = ReportSummary(overall_insight=overall_insight, entity_analyses=entity_analyses)
    else:
        print("[Node: generate_summaries] 구조화된 주장을 찾지 못했습니다. 대체 리포트 요약을 생성합니다.")
        text_summaries = [doc.page_content for doc in state['all_docs'] if doc.metadata.get("element_type") == "text"]
        if text_summaries:
            full_text_context = "\n".join(text_summaries)
            fallback_prompt = PromptTemplate.from_template(
                "다음은 여러 증권사 리포트에서 추출된 핵심 내용 요약본입니다. 이 내용들을 종합하여 리포트 전체의 핵심적인 투자 의견과 전망에 대한 '종합 인사이트'를 3-4 문장으로 작성해주세요.\n\n{context}"
            )
            fallback_chain = fallback_prompt | mini_llm
            overall_insight = (await fallback_chain.ainvoke({"context": full_text_context})).content
            report_summary = ReportSummary(overall_insight=overall_insight, entity_analyses=[])
        else:
            report_summary = None

    # 2. News Summary 생성
    news_summary = None
    all_news = [article for articles_list in state.get('news_articles', {}).values() for article in articles_list]
    if all_news:
        news_summary_chain = build_news_summary_chain(mini_llm)
        generated_part = await news_summary_chain.ainvoke({"news_articles": all_news})
        news_summary = NewsSummary(
            summary=generated_part.summary, key_events=generated_part.key_events, articles=all_news
        )

    return {"report_summary": report_summary, "news_summary": news_summary}


async def generate_grand_summary_node(state: AnalysisState) -> dict:
    """리포트 요약과 뉴스 요약을 합쳐 최종 분석을 생성하는 노드"""
    report_summary = state.get('report_summary')
    news_summary = state.get('news_summary')

    if not report_summary or not news_summary:
        print("[Node: generate_grand_summary] 리포트 또는 뉴스 요약이 없어 최종 분석을 건너뜁니다.")
        return {"grand_summary": None}

    print("[Node: generate_grand_summary] 최종 종합 분석 생성 중...")
    grand_summary_chain = build_grand_summary_chain(state['llm'])
    grand_summary = await grand_summary_chain.ainvoke({
        "report_insight": report_summary.overall_insight,
        "news_summary": news_summary.summary
    })

    return {"grand_summary": grand_summary}


async def run_analysis_workflow(pdf_paths: List[str], pdf_hashes: List[str], mode: str, semaphore: asyncio.Semaphore,
                                llm: BaseChatModel, mini_llm: BaseChatModel):
    workflow = StateGraph(AnalysisState)
    workflow.add_node("process_pdfs", process_pdfs_node)
    workflow.add_node("detect_stocks", detect_stocks_node)
    workflow.add_node("fetch_external_data", fetch_external_data_node)
    workflow.add_node("generate_summaries", generate_summaries_node)
    workflow.add_node("generate_grand_summary", generate_grand_summary_node)

    workflow.add_edge(START, "process_pdfs")
    workflow.add_edge("process_pdfs", "detect_stocks")
    workflow.add_edge("detect_stocks", "fetch_external_data")
    workflow.add_edge("fetch_external_data", "generate_summaries")
    workflow.add_edge("generate_summaries", "generate_grand_summary")
    workflow.add_edge("generate_grand_summary", END)

    app = workflow.compile()
    initial_state = {
        "pdf_paths": pdf_paths, "pdf_hashes": pdf_hashes, "mode": mode,
        "semaphore": semaphore, "llm": llm, "mini_llm": mini_llm
    }
    return await app.ainvoke(initial_state)