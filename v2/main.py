import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import asyncio
import hashlib
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from graph_workflow import run_analysis_workflow
from vector_db import clear_vector_db, load_vector_db_from_disk
from chains import (
    build_preinsight_chain,
    build_global_overview_chain,
    build_final_answer_chain,
    build_final_answer_chain
)
from financial import get_all_tickers
from news_crawler import summarize_news_async
from rag import advanced_rag_search
from dotenv import load_dotenv
from dependencies import get_llm, get_mini_llm, get_reranker

# 환경변수 불러오기
load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# 병렬처리를 위한 동시 요청 수 조정
API_CONCURRENCY_LIMIT = 5
preinsight_cache: dict = {}
global_overview_cache: dict = {} # Global Overview 캐시 추가
analysis_cache: dict = {}
ticker_cache: dict | None = None


@app.on_event("startup")
def startup_event():
    global ticker_cache
    os.makedirs('./tmp', exist_ok=True)
    os.makedirs('./.cache/parsed_pdfs', exist_ok=True)
    os.makedirs('./.cache/summarized_docs', exist_ok=True)
    os.makedirs('./.cache/vector_db', exist_ok=True)

    ticker_cache = get_all_tickers()
    load_vector_db_from_disk()

    print("Pre-loading models...")
    get_llm()
    get_mini_llm()
    get_reranker()
    print("Server startup complete. All resources initialized.")
@app.get("/")
def read_index():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_and_process_pdfs(
        files: List[UploadFile] = File(...),
        mode: str = Form("solar"),
        llm: BaseChatModel = Depends(get_llm),
        mini_llm: BaseChatModel = Depends(get_mini_llm),
):
    clear_vector_db()
    analysis_cache.clear()

    temp_file_paths = []
    pdf_hashes = []
    try:
        for file in files:
            content = await file.read()
            pdf_hash = hashlib.sha256(content).hexdigest()
            file_path = f"./tmp/{file.filename}"
            temp_file_paths.append(file_path)
            pdf_hashes.append(pdf_hash)
            with open(file_path, "wb") as f:
                f.write(content)

        final_state = await run_analysis_workflow(
            pdf_paths=temp_file_paths,
            pdf_hashes=pdf_hashes,
            mode=mode,
            semaphore=asyncio.Semaphore(API_CONCURRENCY_LIMIT),
            llm=llm,
            mini_llm=mini_llm,
        )
        analysis_cache['latest_state'] = final_state
        main_stock_name = list(final_state.get('detected_stocks', {}).keys())[0] if final_state.get(
            'detected_stocks') else "N/A"
        return {"status": "success", "file_count": len(files), "detected_stock": main_stock_name}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {e}")
    finally:
        for file_path in temp_file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)


@app.get("/report-summary")
async def get_report_summary():
    """[신규] UI 요구사항 1: PDF 리포트 요약(증권사별/종목별)만 제공"""
    if 'latest_state' not in analysis_cache:
        return JSONResponse(content={}, status_code=404)
    report_summary = analysis_cache['latest_state'].get('report_summary')
    return JSONResponse(content=report_summary.model_dump() if report_summary else {})


@app.get("/news-summary")
async def get_news_summary():
    """[신규] UI 요구사항 2: 뉴스 요약 정보만 제공"""
    if 'latest_state' not in analysis_cache:
        return JSONResponse(content={}, status_code=404)
    news_summary = analysis_cache['latest_state'].get('news_summary')
    return JSONResponse(content=news_summary.model_dump() if news_summary else {})


@app.get("/financial-chart")
async def get_financial_chart_data():
    """[수정됨] 탐지된 모든 종목의 재무 차트 데이터를 제공하는 API"""
    if 'latest_state' not in analysis_cache:
        return JSONResponse(content={}, status_code=404)

    state = analysis_cache['latest_state']
    financial_data = state.get('financial_data', {})  # { "티커": FinancialTimeSeriesData }
    detected_stocks = state.get('detected_stocks', {})  # { "종목명": "티커" }

    if not financial_data or not detected_stocks:
        return JSONResponse(content=[], status_code=404)

    # 종목명-티커 역방향 맵 생성
    ticker_to_name = {v: k for k, v in detected_stocks.items()}

    response_data = []
    for ticker, ts_data in financial_data.items():
        company_name = ticker_to_name.get(ticker, "N/A")
        response_data.append({
            "company_name": company_name,
            "ticker": ticker,
            "data": ts_data.model_dump()
        })

    return JSONResponse(content=response_data)

@app.post("/ask")
async def ask_question(
        query: str = Form(...),
        llm: BaseChatModel = Depends(get_llm),
        mini_llm: BaseChatModel = Depends(get_mini_llm),
        reranker: HuggingFaceCrossEncoder = Depends(get_reranker)
):
    if 'latest_state' not in analysis_cache:
        raise HTTPException(status_code=400, detail="먼저 PDF 파일을 업로드하고 분석을 완료해주세요.")

    cached_data = analysis_cache['latest_state']
    try:
        rag_docs = await advanced_rag_search(query, k=10, llm=mini_llm, reranker=reranker)

        context_parts = []
        for d in rag_docs:
            source_file = d.metadata.get('source_file', 'N/A')
            page = d.metadata.get('page', 'N/A')
            context_parts.append(f"(출처: {source_file}, 페이지: {page})\n{d.page_content}")
        context = "\n---\n".join(context_parts)

        financial_data = cached_data.get('financial_data', {})
        news_summary_model = cached_data.get('news_summary')
        news_summary_text = f"{news_summary_model.summary}\n주요 이벤트: {', '.join(news_summary_model.key_events)}" if news_summary_model else "최신 뉴스 요약 정보가 없습니다."

        final_answer_chain = build_final_answer_chain(llm)
        analysis_result = await final_answer_chain.ainvoke({
            "question": query,
            "context": context,
            "financial": {k: v.model_dump() for k, v in financial_data.items()},
            "news_summary": news_summary_text,
        })

        return JSONResponse(content=analysis_result.model_dump())
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {e}")