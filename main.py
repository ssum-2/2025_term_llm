import os
import asyncio
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from graph_workflow import run_pdf_workflow
from vector_db import get_vector_db, search_vector_db
from chains import get_report_chain
from financial import get_financial_data
from financial import get_all_tickers
from financial import get_financial_TimeSeries_data
from dotenv import load_dotenv

#
# from chains import get_preinsight_chain

# 환경변수 불러오기
load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# 병렬처리를 위한 동시 요청 수 조정
API_CONCURRENCY_LIMIT = 5

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# asyncio.gather 사용, 비동기
@app.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...), mode: str = Form("solar")):
    tasks = []
    temp_files = []

    # semaphore (공통)
    semaphore = asyncio.Semaphore(API_CONCURRENCY_LIMIT)
    os.makedirs('./tmp', exist_ok=True)
    for file in files:
        file_path = f"./tmp/tmp_{file.filename}"
        temp_files.append(file_path)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        # 각 파일에 대한 워크플로우 실행을 task로 추가
        tasks.append(run_pdf_workflow(file_path, mode, semaphore))

    await asyncio.gather(*tasks)

    # 임시 파일 삭제
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    return {"status": "success", "file_count": len(files)}


@app.post("/ask")
async def ask_question(query: str = Form(...)):
    # dictionary 나중에 빼둘 것
    print(f"[ask_question] query: {query}")
    # stock_dict = {"삼성전자": "005930", "LG에너지솔루션": "373220", "애플": "AAPL", "HDC": "012630", "SK하이닉스": "000660"}
    stock_dict = get_all_tickers()
    matched_stocks = [name for name in stock_dict.keys() if name in query.upper()]
    print(f"[ask_question] matched_stocks: {matched_stocks}")
    ticker = stock_dict.get(matched_stocks[0]) if matched_stocks else "005930"

    print(f"[ask_question] 현재 Vector DB 상태: {get_vector_db()}")

    docs = search_vector_db(query, k=20)  # k는 10~20 정도로 조정; 10 이하는 특정 문서에만 몰리는 현상이 있음
    if not docs:
        return JSONResponse(status_code=404, content={"error": "관련된 내용을 찾을 수 없습니다."})

    print(f"[ask_question] 검색된 문서 수: {len(docs)}")

    context_parts = []
    for doc in docs:
        source = doc.metadata.get('source_file', '알 수 없음')
        page = doc.metadata.get('page', '알 수 없음')
        content = doc.page_content
        context_parts.append(f"--- 문서 출처: {source}, 페이지: {page} ---\n{content}\n")

    context = "\n".join(context_parts)
    # financial = get_financial_data(ticker) if ticker else {}
    financial = get_financial_TimeSeries_data(ticker) if ticker else {}
    chain = get_report_chain()

    try:
        result = await chain.ainvoke({
            "context": context,
            "question": query,
            "financial": financial
        })
        print(f"[ask_question] result: {result}")
        return result
    except Exception as e:
        print(f"[ask_question] result: {result}")
        print(f"[ask_question] ERROR: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "LLM 응답 생성 또는 파싱에 실패했습니다.", "details": str(e)}
        )