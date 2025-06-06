from flask import Flask, request, jsonify, send_from_directory
import os
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import TokenTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import yfinance as yf
from pykrx import stock as pykrx_stock
from datetime import datetime

# 구조화된 출력을 위한 Pydantic 모델
from pydantic import BaseModel, Field
from typing import List


load_dotenv()
app = Flask(__name__)

# 세션별 벡터DB 저장소
session_db = {}

# Upstage Solar LLM 및 임베딩 설정
llm = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model="solar-pro-250422")
embeddings = UpstageEmbeddings(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model="solar-embedding-1-large")

# 리포트 검색 도구
def search_report(query: str, db):
    sub_questions = llm.invoke(f"""
        다음 질문을 3개의 하위 질문으로 분해하세요: {query}
        출력 형식: 1. [하위 질문1], \n2. [하위 질문2], \n3. [하위 질문3]
        """)
    results = []
    print(sub_questions.content)
    for q in sub_questions.content.split("\n"):
        # print(f'q: {q}')
        docs = db.similarity_search(q, k=2)
        results.extend([{
            "content": d.page_content,
            "source": d.metadata.get("source", "알 수 없음"),
            "page": d.metadata.get("page", "알 수 없음")
        } for d in docs])
    return results

# 재무지표 조회 도구 (PyKRX)
def get_pykrx_data(ticker: str, indicator: str):
    print(f'ticker: {ticker}')
    try:
        today = datetime.now().strftime("%Y%m%d")
        if indicator == "price":
            df = pykrx_stock.get_market_ohlcv(today, today, ticker)
            return {"ticker": ticker, "price": df['종가'].iloc[-1]}
        elif indicator == "per":
            df = pykrx_stock.get_market_fundamental(today, today, ticker)
            return {"ticker": ticker, "per": df['PER'].iloc[-1]}
        elif indicator == "pbr":
            df = pykrx_stock.get_market_fundamental(today, today, ticker)
            return {"ticker": ticker, "pbr": df['PBR'].iloc[-1]}
        else:
            return {"error": "지원하지 않는 지표입니다"}
    except Exception as e:
        return {"error": f"PyKRX 데이터 가져오기 실패: {str(e)}"}

# 재무지표 조회 도구 (Yahoo Finance)
def get_yahoo_finance(ticker: str, indicator: str):
    try:
        if indicator == "price":
            data = yf.Ticker(ticker).history(period="1d")
            return {"ticker": ticker, "price": data['Close'].iloc[-1]}
        elif indicator == "pe_ratio":
            info = yf.Ticker(ticker).info
            return {"ticker": ticker, "per": info.get('trailingPE', 'N/A')}
        elif indicator == "dividend":
            info = yf.Ticker(ticker).info
            return {"ticker": ticker, "dividend": info.get('dividendYield', 'N/A') * 100}
        else:
            return {"error": "지원하지 않는 지표입니다"}
    except Exception as e:
        return {"error": f"야후 파이낸스 데이터 가져오기 실패: {str(e)}"}

# PDF 업로드
@app.route('/upload', methods=['POST'])
def upload_pdf():
    files = request.files.getlist('pdfs')
    session_id = request.form.get('session_id', 'default')
    texts = []
    for file in files:
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages[1:]):
            text = page.extract_text()
            texts.append((text, {"source": file.filename, "page": i+3}))
    splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=50)
    docs = []
    for text, metadata in texts:
        chunks = splitter.create_documents([text], [metadata])
        docs.extend(chunks)
    db = FAISS.from_documents(docs, embeddings)
    session_db[session_id] = db
    return jsonify({"status": "success", "session_id": session_id})

class Company(BaseModel):
    name: str = Field(..., description="증권사 이름")
    opinion: str = Field(..., description="의견 (매수/중립/매도 등)")
    target_price: str = Field(..., description="목표주가")
    reason: str = Field(..., description="근거")
    source: str = Field(..., description="출처 (리포트명, 페이지)")

class FinancialData(BaseModel):
    ticker: str = Field(..., description="종목 코드")
    price: float = Field(..., description="주가")
    per: float = Field(..., description="PER")
    pbr: float = Field(..., description="PBR")

class FinalAnswer(BaseModel):
    summary: str = Field(..., description="요약 내용")
    companies: List[Company] = Field(..., description="증권사별 의견")
    financial_data: FinancialData = Field(..., description="재무지표")

# Function Calling 방식으로 질문 처리
@app.route('/ask', methods=['POST'])
def ask_question():
    session_id = request.form.get('session_id', 'default')
    query = request.form.get('query')

    if session_id not in session_db:
        return jsonify({"error": "세션이 존재하지 않습니다."})

    db = session_db[session_id]

    # 종목명-코드 사전 (현재는 사전에서 매칭된 코드로 조회하도록 함. dictionary를 외부파일로 빼는 등 처리 필요)
    stock_dict = {
        "삼성전자": "005930",
        "LG에너지솔루션": "373220",
        "애플": "AAPL",
        "HDC": "012630"
    }

    # 종목명 추출
    matched_stocks = [name for name in stock_dict.keys() if name in query]
    if matched_stocks:
        ticker = stock_dict[matched_stocks[0]]  # 첫 번째로 매칭된 종목 사용
    else:
        ticker = "005930"  # 기본값

    # 리포트 검색
    report_results = search_report(query, db)
    print(f'report_results: {report_results}')
    summary = " ".join([r["content"] for r in report_results])
    if len(summary) > 1000:
        summary = summary[:1000] + "..."
    companies = []
    for r in report_results:
        companies.append({
            "name": r["source"],
            "opinion": "알 수 없음",
            "target_price": "알 수 없음",
            "reason": r["content"],
            "source": f"{r['source']} p.{r['page']}"
        })

    # 재무지표 조회 (동적으로 ticker 사용)
    financial_data = get_pykrx_data(ticker, "price")
    if "error" not in financial_data:
        per = get_pykrx_data(ticker, "per")
        pbr = get_pykrx_data(ticker, "pbr")
        if "error" not in per and "error" not in pbr:
            financial_data["per"] = per["per"]
            financial_data["pbr"] = pbr["pbr"]
        else:
            financial_data["per"] = "알 수 없음"
            financial_data["pbr"] = "알 수 없음"
    else:
        financial_data = {"ticker": ticker, "price": "알 수 없음", "per": "알 수 없음", "pbr": "알 수 없음"}

    print(f'financial_data: {financial_data}')
    # LLM에 리포트 요약, 증권사별 의견, 재무지표를 전달하고 JSON 형식으로 답변 요청
    prompt = f"""
    [Role] 당신은 증권 애널리스트 리포트 분석 전문가 AI입니다.
    [Rules]
    1. 아래 리포트 요약, 증권사별 의견, 재무지표를 바탕으로 답변하세요.
    2. 답변은 반드시 아래 JSON 형식으로만 출력하세요.

    리포트 요약: {summary}
    증권사별 의견: {companies}
    재무지표: {financial_data}

    JSON 형식 예시:
    {{
      "summary": "요약 내용",
      "companies": [
        {{
          "name": "A증권",
          "opinion": "매수",
          "target_price": "95,000원",
          "reason": "반도체 산업 성장 전망",
          "source": "A증권 리포트 p.12"
        }},
        {{
          "name": "B증권",
          "opinion": "중립",
          "target_price": "82,000원",
          "reason": "실적 호조, 시장 변동성 주의",
          "source": "B증권 리포트 p.7"
        }}
      ],
      "financial_data": {{
        "ticker": "{ticker}",
        "price": {financial_data.get('price', '알 수 없음')},
        "per": {financial_data.get('per', '알 수 없음')},
        "pbr": {financial_data.get('pbr', '알 수 없음')}
      }}
    }}
    """

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    print(f'response: {response.content}')
    try:
        start = response.content.find('{')
        end = response.content.rfind('}')
        if start == -1 or end == -1:
            return jsonify({"error": "JSON 형식이 아닙니다.", "raw_output": response.content})
        json_str = response.content[start:end+1]
        result = json.loads(json_str)
        print(f'final_result: {result}')
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"JSON 파싱 실패: {str(e)}", "raw_output": response.content})


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8887', debug=True)
