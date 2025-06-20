import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import asyncio, hashlib, traceback, shutil # <-- shutil 임포트 추가
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from graph_workflow import run_analysis_workflow
from vector_db import clear_vector_db, load_vector_db_from_disk
from chains import build_final_answer_chain
from financial import get_all_tickers
from news_crawler import get_latest_news
from rag import advanced_rag_search
from dotenv import load_dotenv
from dependencies import get_llm, get_mini_llm, get_reranker

load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

API_CONCURRENCY_LIMIT = 5
analysis_cache: Dict[str, Any] = {}
ticker_cache: dict | None = None


@app.on_event("startup")
def startup_event():
    global ticker_cache
    for dir_path in ['./tmp', './.cache/parsed_pdfs', './.cache/summarized_docs', './.cache/vector_db', './tmp/images']:
        os.makedirs(dir_path, exist_ok=True)
    ticker_cache = get_all_tickers()
    load_vector_db_from_disk()
    get_llm();
    get_mini_llm();
    get_reranker()
    print("Server startup complete.")

@app.get("/")
def read_index(): return FileResponse("static/index.html")


@app.post("/upload")
async def upload_and_process_pdfs(
        files: List[UploadFile] = File(...),
        mode: str = Form("solar"),
        llm: BaseChatModel = Depends(get_llm),
        mini_llm: BaseChatModel = Depends(get_mini_llm),
):
    if os.path.exists("./.cache/summarized_docs"):
        shutil.rmtree("./.cache/summarized_docs")
    os.makedirs("./.cache/summarized_docs", exist_ok=True)
    clear_vector_db()  # Vector DB 캐시도 함께 초기화

    analysis_cache.clear()

    temp_file_paths, pdf_hashes = [], []
    try:
        for file in files:
            content = await file.read()
            pdf_hashes.append(hashlib.sha256(content).hexdigest())
            path = f"./tmp/{file.filename}"
            temp_file_paths.append(path)
            with open(path, "wb") as f: f.write(content)

        final_state = await run_analysis_workflow(
            pdf_paths=temp_file_paths, pdf_hashes=pdf_hashes, mode=mode,
            semaphore=asyncio.Semaphore(API_CONCURRENCY_LIMIT), llm=llm, mini_llm=mini_llm
        )
        analysis_cache['latest_state'] = final_state
        detected_stocks = final_state.get('detected_stocks', {})
        return {"status": "success", "file_count": len(files),
                "detected_stock": ", ".join(detected_stocks.keys()) or "탐지된 종목 없음"}
    except Exception as e:
        tb_str = traceback.format_exc()
        print("--- DETAILED ERROR TRACEBACK ---")
        print(tb_str)
        print("---------------------------------")
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {e}")
    finally:
        for file_path in temp_file_paths:
            if os.path.exists(file_path): os.remove(file_path)



@app.get("/grand-summary")
async def get_grand_summary():
    if 'latest_state' not in analysis_cache:
        return JSONResponse(content={}, status_code=404)
    grand_summary = analysis_cache['latest_state'].get('grand_summary')
    return JSONResponse(content=grand_summary.model_dump() if grand_summary else {})


@app.get("/report-summary")
async def get_report_summary():
    if 'latest_state' not in analysis_cache:
        return JSONResponse(content={}, status_code=404)
    report_summary = analysis_cache['latest_state'].get('report_summary')
    return JSONResponse(content=report_summary.model_dump() if report_summary else {})


@app.get("/news-summary")
async def get_news_summary():
    if 'latest_state' not in analysis_cache:
        return JSONResponse(content={}, status_code=404)
    news_summary = analysis_cache['latest_state'].get('news_summary')
    return JSONResponse(content=news_summary.model_dump() if news_summary else {})


@app.get("/financial-chart")
async def get_financial_chart_data():
    if 'latest_state' not in analysis_cache:
        return JSONResponse(content=[], status_code=404)

    state = analysis_cache['latest_state']
    financial_data = state.get('financial_data', {})
    detected_stocks = state.get('detected_stocks', {})

    # 재무 데이터나 탐지된 종목이 없으면 404가 아닌 빈 리스트를 반환하여 프론트엔드 에러 방지
    if not financial_data or not detected_stocks:
        return JSONResponse(content=[])

    ticker_to_name = {v: k for k, v in detected_stocks.items()}
    response_data = []
    for ticker, ts_data in financial_data.items():
        company_name = ticker_to_name.get(ticker, "N/A")
        response_data.append({"company_name": company_name, "ticker": ticker, "data": ts_data.model_dump()})
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

    state = analysis_cache['latest_state']
    try:
        rag_docs = await advanced_rag_search(query, k=15, llm=mini_llm, reranker=reranker)
        context = "\n---\n".join(
            [f"(출처: {d.metadata.get('source_file', 'N/A')}, 페이지: {d.metadata.get('page', 'N/A')})\n{d.page_content}" for
             d in rag_docs])

        financial_data = state.get('financial_data', {})
        news_summary_model = state.get('news_summary')
        news_summary_text = news_summary_model.summary if news_summary_model else "최신 뉴스 요약 정보가 없습니다."

        final_answer_chain = build_final_answer_chain(llm)
        analysis_result = await final_answer_chain.ainvoke({
            "question": query, "context": context,
            "financial": {k: v.model_dump() for k, v in financial_data.items()},
            "news_summary": news_summary_text,
        })
        return JSONResponse(content=analysis_result.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {e}")
