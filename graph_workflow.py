import asyncio
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from vector_db import save_to_vector_db
from utils import parse_pdf, summarize_single_element



class PDFState(TypedDict):
    pdf_path: str
    mode: str
    semaphore: asyncio.Semaphore
    elements: List[dict]
    summaries: List[Document]


def parse_pdf_node(state: PDFState) -> dict:
    print(f"[Node: parse_pdf] PDF 파싱 시작: {state['pdf_path']}")
    elements = parse_pdf(state['pdf_path'])
    print(f"[Node: parse_pdf] 파싱 완료, {len(elements)}개 요소 발견")
    return {"elements": elements}


# 병렬처리
async def parallel_summarize_node(state: PDFState) -> dict:
    mode = state['mode']
    pdf_path = state['pdf_path']
    elements = state['elements']
    semaphore = state['semaphore']

    print(f"[Node: parallel_summarize] {len(elements)}개 요소 병렬 요약 시작...")

    async def summarize_with_semaphore(element):
        async with semaphore:
            return await asyncio.to_thread(
                summarize_single_element,
                {**element, "mode": mode, "pdf_path": pdf_path}
            )

    tasks = [summarize_with_semaphore(element) for element in elements]
    summary_docs = await asyncio.gather(*tasks)
    valid_docs = [doc for doc in summary_docs if doc is not None and isinstance(doc, Document)]

    print(f"[Node: parallel_summarize] 병렬 요약 완료, {len(valid_docs)}개 요약 생성됨.")
    return {"summaries": valid_docs}

# 비동기 처리
async def embedding_node(state: PDFState) -> dict:
    print(f"[Node: embedding] 벡터 DB 저장 시작...")
    summaries_to_embed = state.get("summaries", [])
    if not summaries_to_embed:
        print("[Node: embedding] 저장할 요약이 없습니다. 건너뜁니다.")
        return {}

    await save_to_vector_db(summaries_to_embed)
    print(f"[Node: embedding] 벡터 DB 저장 완료.")
    return {}


async def run_pdf_workflow(pdf_path: str, mode: str, semaphore: asyncio.Semaphore):
    # StateGraph 정의
    workflow = StateGraph(PDFState)

    # 노드 추가
    workflow.add_node("parse_pdf", parse_pdf_node)
    workflow.add_node("parallel_summarize", parallel_summarize_node)
    workflow.add_node("embed", embedding_node)

    # 엣지 연결 (직선 그래프)
    workflow.add_edge(START, "parse_pdf")
    workflow.add_edge("parse_pdf", "parallel_summarize")
    workflow.add_edge("parallel_summarize", "embed")
    workflow.add_edge("embed", END)

    # 그래프 컴파일
    app = workflow.compile()

    # 실행
    inputs = {"pdf_path": pdf_path, "mode": mode, "semaphore": semaphore}
    await app.ainvoke(inputs)

    print(f"--- 워크플로우 완료: {pdf_path} ---")

