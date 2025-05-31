from flask import Flask, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_upstage import ChatUpstage
from langchain.agents import tool, AgentExecutor, Tool, create_react_agent
from langchain import hub
# from lmnr import Laminar

app = Flask(__name__)

# global
# 세션별 벡터DB 저장소
session_db = {}

load_dotenv()
llm = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model="solar-pro-250422")
custom_react_prompt = hub.pull("hwchase17/react").partial(
    system_message="""
    [Role] 당신은 공부 도우미 전문가 AI입니다.
    [Rules]
    1. 모든 답변은 강의자료 검색 결과를 기반으로 해야 함
    2. 복잡한 개념은 반드시 실제 사례 포함
    3. 답변은 5문장 이내로 간결하게 작성
    4. 단계별 추론 과정을 명확히 표시
    5. "Final Answer:"가 나오면 절대 "Action:"을 추가하지 마세요.
    6. "For troubleshooting, visit: ..." 같은 불필요한 문장을 추가하지 마세요.
    7. 요약을 요청하는 경우 구조화된 형식으로 출력
    8. 자세한 설명을 요청하는 경우 10문장까지 확장하고 목록형으로 구조화하여 작성

    [Step-by-Step Process]
    1. Thought: 문제 분석 및 해결 계획 수립
    2. Action: 강의자료 검색 도구 실행
    3. Observation: 검색 결과 분석
    4. Thought: 최종 답변 설계
    5. Final Answer: 규칙 준수한 답변 제공
    """
)
#laminar 추적
# Laminar.initialize(project_api_key=os.getenv('LMNR_PROJECT_API_KEY'))

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# PDF 파일 업로드
@app.route('/upload', methods=['POST'])
def upload_pdf():
    files = request.files.getlist('pdfs')
    session_id = request.form.get('session_id', 'default')

    texts = []
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages[2:]:  # 표지, 목차 제거
            texts.append(page.extract_text())

    # 텍스트 청킹
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts)

    # 임베딩 및 벡터DB 저장
    embeddings = UpstageEmbeddings(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model="solar-embedding-1-large")
    db = FAISS.from_documents(docs, embeddings)
    session_db[session_id] = db

    return jsonify({"status": "success", "session_id": session_id})


# 사용자 질의
@app.route('/ask', methods=['POST'])
def ask_question():
    session_id = request.form.get('session_id', 'default')
    query = request.form.get('query')

    if session_id not in session_db:
        return jsonify({"error": "세션이 존재하지 않습니다."})

    db = session_db[session_id]

    # PDF 검색 툴, closure
    @tool
    def search_pdf(query: str) -> str:
        """강의자료(PDF)에서 관련 내용을 검색하여 반환합니다."""
        sub_questions = llm.invoke(f"""
            다음 질문을 3개의 하위 질문으로 분해하세요: {query}
            출력 형식: 1. [하위 질문1], 2. [하위 질문2], 3. [하위 질문3]
            """)
        results = []
        for q in sub_questions.split("\n"):
            docs = db.similarity_search(q, k=2)
            results.extend([d.page_content for d in docs])
        return "\n\n".join(results[:5])

    tools = [Tool(name="search_pdf", func=search_pdf, description="강의자료(PDF)에서 관련 내용을 검색")]

    agent = create_react_agent(llm, tools, custom_react_prompt)
    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   verbose=True,
                                   handle_parsing_errors=True,
                                   max_iterations=5)

    result = agent_executor.invoke({"input": query})
    return jsonify({"answer": result["output"]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8887', debug=False)
