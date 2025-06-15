# local_test.py (수정)
import asyncio
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dotenv import load_dotenv

load_dotenv()
from dependencies import get_llm, get_mini_llm, get_reranker, get_openai_client
from utils import parse_pdf
from chains import build_preinsight_chain
from financial import get_financial_timeseries_data_for_llm, get_all_tickers
from news_crawler import get_latest_news, summarize_news_async
from rag import advanced_rag_search
from vector_db import save_to_vector_db, clear_vector_db, search_vector_db

# --- 테스트 설정 ---
TEST_PDF_PATH = "./AnalReports/samsung_report.pdf"  # 👈 테스트할 PDF 경로
TEST_STOCK_NAME = "삼성전자"
TEST_TICKER = "005930"
TEST_QUERY = "삼성전자의 HBM 사업 전망과 리스크는 무엇인가?"


# -----------------

async def test_01_pdf_processing():
    print("\n--- 🧪 테스트 1: PDF 처리 (요약+정보추출) 및 임베딩 ---")
    if not os.path.exists(TEST_PDF_PATH):
        print(f"'{TEST_PDF_PATH}' 파일이 없습니다. 테스트를 건너뜁니다.")
        return

    # 테스트를 위해 모델 직접 로드
    mini_llm = get_mini_llm()
    openai_client = get_openai_client()

    clear_vector_db()
    from graph_workflow import _summarize_elements_in_parallel

    elements = parse_pdf(TEST_PDF_PATH)
    state = {
        "elements": elements,
        "pdf_path": TEST_PDF_PATH,
        "mode": os.getenv("LLM_MODE", "solar"),
        "semaphore": asyncio.Semaphore(5),
        "processed_docs": [],
        "llm": mini_llm,  # 모델 전달
        "openai_client": openai_client  # 모델 전달
    }

    processed_docs = await _summarize_elements_in_parallel(state)
    print(f"\n[처리 결과] 총 {len(processed_docs)}개의 문서/정보 생성")

    summary_count = sum(1 for d in processed_docs if d.metadata.get('element_type') != 'broker_info')
    broker_info_count = sum(1 for d in processed_docs if d.metadata.get('element_type') == 'broker_info')
    print(f"  - 요약 문서: {summary_count}개")
    print(f"  - 추출된 증권사 정보: {broker_info_count}개")

    if broker_info_count > 0:
        print("\n[추출된 정보 예시]")
        for doc in processed_docs:
            if doc.metadata.get('element_type') == 'broker_info':
                print(doc.page_content)
                break

    await save_to_vector_db(processed_docs)


async def test_02_financial_data_and_news():
    print(f"\n--- 🧪 테스트 2: '{TEST_STOCK_NAME}' 재무 데이터 및 뉴스 조회 ---")
    financial_data = get_financial_timeseries_data_for_llm(TEST_TICKER, years=1)
    print("\n[재무 데이터 (최근 3개월)]")
    if financial_data and financial_data.price:
        price_dict = financial_data.model_dump()['price']
        per_dict = financial_data.model_dump()['per']
        pbr_dict = financial_data.model_dump()['pbr']
        for date in sorted(list(price_dict.keys()))[-3:]:
            print(f"  - {date}: 주가={price_dict.get(date)}, PER={per_dict.get(date)}, PBR={pbr_dict.get(date)}")
    else:
        print("재무 데이터를 가져오지 못했습니다.")

    articles = await asyncio.to_thread(get_latest_news, TEST_STOCK_NAME)
    print("\n[최신 뉴스 기사]")
    if articles:
        for article in articles[:2]:
            print(f"- {article['title']} ({article['publisher']})")

        mini_llm = get_mini_llm()
        news_summary = await summarize_news_async(articles, mini_llm)
        print("\n[뉴스 요약]")
        print(news_summary)
    else:
        print("최신 뉴스가 없습니다.")


async def test_03_advanced_rag_pipeline():
    print("\n--- 🧪 테스트 3: 고도화된 RAG 파이프라인 (분리 검색) ---")
    # Vector DB에 데이터가 있는지 확인하는 로직 수정 (get_all_tickers는 KRX 종목 목록을 가져오므로 부적절)
    try:
        if not search_vector_db("test", k=1):
            print("Vector DB가 비어있어 RAG 테스트를 건너뜁니다. 먼저 test_01_pdf_processing을 실행하세요.")
            return
    except Exception:
        print("Vector DB가 비어있어 RAG 테스트를 건너뜁니다. 먼저 test_01_pdf_processing을 실행하세요.")
        return

    print(f"\n[RAG 테스트] 질문: {TEST_QUERY}")

    mini_llm = get_mini_llm()
    reranker = get_reranker()

    general_docs = await advanced_rag_search(TEST_QUERY, k=5, llm=mini_llm, reranker=reranker)
    print(f"\n[일반 컨텍스트 검색 결과 {len(general_docs)}개]")
    for i, doc in enumerate(general_docs):
        print(f"  - Doc {i + 1}: {doc.page_content[:80]}...")

    broker_docs = search_vector_db(TEST_QUERY, k=5, filter_criteria={"element_type": "broker_info"})
    print(f"\n[증권사 의견 검색 결과 {len(broker_docs)}개]")
    for i, doc in enumerate(broker_docs):
        print(f"  - Card {i + 1}: {doc.page_content.replace(chr(10), ' ')}")


async def test_04_preinsight_chain():
    print("\n--- 🧪 테스트 4: Pre-insight 체인 (구조화 데이터 기반) ---")
    broker_docs = search_vector_db("", k=15, filter_criteria={"element_type": "broker_info"})
    if not broker_docs:
        print("추출된 증권사 정보가 없어 Pre-insight 테스트를 건너뜁니다.")
        return

    context = "\n---\n".join([d.page_content for d in broker_docs])
    financial_data = get_financial_timeseries_data_for_llm(TEST_TICKER).model_dump()

    articles = await asyncio.to_thread(get_latest_news, TEST_STOCK_NAME, max_results=5)
    news_titles = "\n".join([f"- {a['title']}" for a in articles]) if articles else "최신 뉴스가 없습니다."

    mini_llm = get_mini_llm()
    chain = build_preinsight_chain(mini_llm)
    insight = await chain.ainvoke({
        "context": context,
        "financial": financial_data,
        "news_titles": news_titles
    })
    print("\n[Pre-insight 생성 결과]")
    print("하이라이트:")
    for h in insight.highlights:
        print(f"- {h}")
    print("\n추천 질문:")
    for q in insight.suggested_questions:
        print(f"- {q}")


async def main():
    print("KRX 티커 목록 로딩 중...")
    get_all_tickers()

    # 테스트에 필요한 모델 미리 로딩
    print("Pre-loading models for test...")
    get_llm()
    get_mini_llm()
    get_reranker()
    get_openai_client()

    await test_01_pdf_processing()
    await test_02_financial_data_and_news()
    await test_03_advanced_rag_pipeline()
    await test_04_preinsight_chain()


if __name__ == "__main__":
    if not os.path.exists("./AnalReports"):
        os.makedirs("./AnalReports")
        print("경고: './AnalReports' 폴더가 없어 새로 생성했습니다. 테스트를 위해 해당 폴더에 PDF 파일을 넣어주세요.")

    asyncio.run(main())