import asyncio, os, re
from typing import TypedDict, List, Dict, Optional
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END, MessageGraph

from dependencies import get_openai_client
from vector_db import save_to_vector_db
from utils import PDFSplitter, LayoutAnalyzer, ElementProcessor
from chains import (
    build_extractor_chain, build_entity_summary_chain,
    build_news_summary_chain, build_grand_summary_chain,
    ReportSummary, NewsSummary, EntityAnalysis, GrandSummary
)
from financial import get_financial_timeseries_data_for_llm, FinancialTimeSeriesData
from news_crawler import get_latest_news


class AnalysisState(TypedDict):
    pdf_paths: List[str]
    mode: str
    llm: BaseChatModel
    mini_llm: BaseChatModel

    # PDF 파싱을 위한 상세 상태
    split_pdf_paths: Dict[str, List[str]]
    analysis_results: Dict[str, Dict]
    structured_elements: Dict[str, Dict]

    # 최종 결과물
    all_docs: List[Document]
    detected_stocks: Dict[str, str]
    financial_data: Dict[str, FinancialTimeSeriesData]
    news_articles: Dict[str, List[dict]]
    report_summary: Optional[ReportSummary]
    news_summary: Optional[NewsSummary]
    grand_summary: Optional[GrandSummary]


async def split_pdf_node(state: AnalysisState) -> dict:
    """PDF를 작은 조각으로 분할하는 노드"""
    print("--- 1. PDF 분할 노드 시작 ---")
    split_pdf_paths = {}
    for path in state["pdf_paths"]:
        split_pdf_paths[path] = PDFSplitter.split(path)
    return {"split_pdf_paths": split_pdf_paths}


async def layout_analysis_node(state: AnalysisState) -> dict:
    """분할된 PDF 조각들의 레이아웃을 분석하는 노드"""
    print("--- 2. Layout 분석 노드 시작 ---")
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    if not upstage_api_key: raise ValueError("UPSTAGE_API_KEY가 .env에 없습니다.")

    analyzer = LayoutAnalyzer(upstage_api_key)
    analysis_results = {}

    semaphore = asyncio.Semaphore(5)
    async def analyze_file(path):
        # 세마포를 사용하여 동시 실행 수를 제어
        async with semaphore:
            print(f"  -> Layout 분석 시작: {os.path.basename(path)}")
            # analyzer.analyze는 동기 함수이므로 to_thread로 비동기 실행
            result = await asyncio.to_thread(analyzer.analyze, path)
            print(f"  <- Layout 분석 완료: {os.path.basename(path)}")
            return path, result

    tasks = [analyze_file(p) for paths in state["split_pdf_paths"].values() for p in paths]
    results = await asyncio.gather(*tasks)

    for path, result in results:
        if result: analysis_results[path] = result

    # 오류가 발생한 결과를 명시적으로 출력
    failed_analyses = [path for path, res in results if not res]
    if failed_analyses:
        print(f"❌ Layout 분석 실패 목록 ({len(failed_analyses)}개):")
        for f in failed_analyses:
            print(f"  - {os.path.basename(f)}")

    return {"analysis_results": analysis_results}


async def process_elements_node(state: AnalysisState) -> dict:
    """
    레이아웃 분석 결과를 바탕으로 '페이지 단위'로 요소를 그룹화하고,
    요약 및 주장 추출을 수행하여 API 호출을 최소화하는 노드
    """
    print("--- 3. 요소 처리 및 문서 생성 노드 시작 (비용 최적화 버전) ---")
    llm, mini_llm = state['llm'], state['mini_llm']
    openai_client = get_openai_client()
    all_docs = []

    for original_path, split_paths in state["split_pdf_paths"].items():
        source_file = os.path.basename(original_path)
        relevant_analysis = {p: state["analysis_results"][p] for p in split_paths if p in state["analysis_results"]}
        if not relevant_analysis: continue

        page_elements = ElementProcessor.extract_and_structure_elements(relevant_analysis)
        author_match = re.search(r'([\w\s]+(?:증권|투자증권|자산운용|경제연구소|리서치))',
                                 list(relevant_analysis.values())[0].get('html', ''))
        author = author_match.group(1).strip().replace("\n", " ") if author_match else "작성자"

        print(f"📄 [{source_file}] 저자: '{author}', 페이지: {len(page_elements)}")

        ElementProcessor.crop_and_save_elements(original_path, page_elements)

        # --- 수정된 핵심 로직: 페이지 단위 처리 ---
        page_content_for_extraction = []
        for page_num, elements in page_elements.items():
            # 1. 한 페이지의 모든 텍스트 요소를 결합
            page_text = "\n".join([e.get('text', '') for e in elements if e.get('category') in ['title', 'header', 'paragraph', 'list']])

            # 2. (선택적) 이미지/테이블 요약 (기존 로직 유지하되, 비용 문제 시 비활성화 가능)
            multimodal_summaries = []
            mm_tasks = [
                asyncio.to_thread(ElementProcessor.summarize_element, e, page_text, mini_llm, openai_client)
                for e in elements if e.get('category') in ['figure', 'table']
            ]
            if mm_tasks:
                summaries = await asyncio.gather(*mm_tasks)
                multimodal_summaries = list(filter(None, summaries))

            # 3. 텍스트와 이미지/테이블 요약을 합쳐 페이지 전체 요약본 생성
            full_page_content = page_text
            if multimodal_summaries:
                full_page_content += "\n\n[이미지/표 요약]\n" + "\n".join(multimodal_summaries)

            page_content_for_extraction.append(full_page_content)
            all_docs.append(Document(page_content=full_page_content, metadata={"source_file": source_file, "page": page_num,
                                                                          "element_type": "page_summary"}))

        # --- 주장 추출: 전체 리포트 내용을 한 번에 처리 ---
        full_report_summary = "\n\n".join(page_content_for_extraction)
        if full_report_summary:
            extractor_chain = build_extractor_chain(llm)
            try:
                # 리포트 전체 텍스트를 한 번에 넣어 주장 추출 (API 호출 1회)
                extracted_data = await extractor_chain.ainvoke({"text": full_report_summary, "source_entity_name": author})
                assertions = extracted_data.assertions
                print(f"✅ [{source_file}] 주장 추출: {len(assertions)}개")
                for assertion in assertions:
                    all_docs.append(Document(
                        page_content=f"주장: {assertion.assertion}\n근거: {assertion.evidence}",
                        metadata={"source_file": source_file, "page": "N/A", "element_type": "assertion_info",
                                  "source_entity": author, "subject": assertion.subject,
                                  "sentiment": assertion.sentiment}
                    ))
            except Exception as e:
                print(f"❌ [{source_file}] 주장 추출 실패: {e}")

    await save_to_vector_db(all_docs)
    return {"all_docs": all_docs}

async def detect_stocks_node(state: AnalysisState) -> dict:
    from main import ticker_cache
    if not ticker_cache: return {"detected_stocks": {}}

    stock_counts = defaultdict(int)
    all_docs = state['all_docs']

    stock_names_sorted = sorted(ticker_cache.keys(), key=len, reverse=True)

    for path in state['pdf_paths']:
        filename = os.path.basename(path)
        for stock_name in stock_names_sorted:
            if stock_name in filename:
                stock_counts[stock_name] += 50

    assertion_docs = [doc for doc in all_docs if doc.metadata.get("element_type") == "assertion_info"]
    for doc in assertion_docs:
        subject = doc.metadata.get('subject', '')
        for stock_name in stock_names_sorted:
            if stock_name in subject:
                stock_counts[stock_name] += 10

    text_docs_content = " ".join(
        [doc.page_content for doc in all_docs if doc.metadata.get("element_type") == "page_summary"])
    temp_text_content = text_docs_content
    for stock_name in stock_names_sorted:
        count = temp_text_content.count(stock_name)
        if count > 0:
            stock_counts[stock_name] += count
            temp_text_content = temp_text_content.replace(stock_name, "")

    valid_counts = {
        name: score for name, score in stock_counts.items()
        if "증권" not in name and "자산운용" not in name and "투자" not in name and "금융" not in name and score > 0
    }

    if not valid_counts:
        print("️⚠️ [Node: detect_stocks] 탐지된 종목 없음.")
        return {"detected_stocks": {}}

    top_stocks = sorted(valid_counts.items(), key=lambda item: item[1], reverse=True)[:3]
    detected_stocks = {name: ticker_cache[name] for name, count in top_stocks}
    print(f"✅ [Node: detect_stocks] 최종 탐지된 종목: {list(detected_stocks.keys())}")
    return {"detected_stocks": detected_stocks}


async def fetch_external_data_node(state: AnalysisState) -> dict:
    detected_stocks = state.get('detected_stocks', {})
    if not detected_stocks:
        print("️⚠️ [Node: fetch_external_data] 탐지된 종목이 없어 외부 데이터 수집을 건너뜁니다.")
        return {"financial_data": {}, "news_articles": {}}

    financial_tasks = {name: asyncio.to_thread(get_financial_timeseries_data_for_llm, ticker) for name, ticker in
                       detected_stocks.items()}
    news_tasks = {name: asyncio.to_thread(get_latest_news, name, max_results=5) for name in detected_stocks.keys()}

    financial_results, news_results = await asyncio.gather(
        asyncio.gather(*financial_tasks.values()),
        asyncio.gather(*news_tasks.values())
    )

    financial_data = {list(detected_stocks.keys())[i]: data for i, data in enumerate(financial_results) if
                      data and data.price}
    news_articles = {list(detected_stocks.keys())[i]: news for i, news in enumerate(news_results) if news}

    return {"financial_data": financial_data, "news_articles": news_articles}


async def generate_summaries_node(state: AnalysisState) -> dict:
    mini_llm = state['mini_llm']
    assertion_docs = [doc for doc in state['all_docs'] if doc.metadata.get("element_type") == "assertion_info"]

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
        insight_context = "\n\n".join(f"### {a.entity_name}\n- {a.main_stance}" for a in entity_analyses)
        insight_prompt = PromptTemplate.from_template(
            "다음은 여러 분석 주체들의 핵심 입장입니다. 이를 종합하여 전체 시장과 분석 대상에 대한 최종 인사이트를 3-4 문장으로 요약해주세요.\n\n{context}")
        overall_insight = (await (insight_prompt | mini_llm).ainvoke({"context": insight_context})).content
        report_summary = ReportSummary(overall_insight=overall_insight, entity_analyses=entity_analyses)
    else:
        text_summaries = [doc.page_content for doc in state['all_docs'] if
                          doc.metadata.get("element_type") == "page_summary"]
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

    news_summary = None
    all_news = [article for articles_list in state.get('news_articles', {}).values() for article in articles_list]
    if all_news:
        news_summary_chain = build_news_summary_chain(mini_llm)
        generated_part = await news_summary_chain.ainvoke({"news_articles": all_news})
        news_summary = NewsSummary(summary=generated_part.summary, key_events=generated_part.key_events,
                                   articles=all_news)

    return {"report_summary": report_summary, "news_summary": news_summary}


async def generate_grand_summary_node(state: AnalysisState) -> dict:
    report_summary = state.get('report_summary')
    news_summary = state.get('news_summary')

    if not report_summary or not news_summary:
        return {"grand_summary": None}

    grand_summary_chain = build_grand_summary_chain(state['llm'])
    grand_summary = await grand_summary_chain.ainvoke({
        "report_insight": report_summary.overall_insight,
        "news_summary": news_summary.summary
    })

    return {"grand_summary": grand_summary}


def build_graph():
    """LangGraph 워크플로우를 구성하고 컴파일합니다."""
    workflow = StateGraph(AnalysisState)

    workflow.add_node("split_pdfs", split_pdf_node)
    workflow.add_node("layout_analysis", layout_analysis_node)
    workflow.add_node("process_elements", process_elements_node)
    workflow.add_node("detect_stocks", detect_stocks_node)
    workflow.add_node("fetch_external_data", fetch_external_data_node)
    workflow.add_node("generate_summaries", generate_summaries_node)
    workflow.add_node("generate_grand_summary", generate_grand_summary_node)

    workflow.set_entry_point("split_pdfs")
    workflow.add_edge("split_pdfs", "layout_analysis")
    workflow.add_edge("layout_analysis", "process_elements")
    workflow.add_edge("process_elements", "detect_stocks")
    workflow.add_edge("detect_stocks", "fetch_external_data")
    workflow.add_edge("fetch_external_data", "generate_summaries")
    workflow.add_edge("generate_summaries", "generate_grand_summary")
    workflow.add_edge("generate_grand_summary", END)

    return workflow.compile()


app = build_graph()
async def run_analysis_workflow(pdf_paths: List[str], mode: str, llm: BaseChatModel, mini_llm: BaseChatModel):
    initial_state = {
        "pdf_paths": pdf_paths,
        "mode": mode,
        "llm": llm,
        "mini_llm": mini_llm,
    }
    return await app.ainvoke(initial_state)
