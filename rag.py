import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.language_models.chat_models import BaseChatModel
from vector_db import get_vector_db

async def get_hyde_documents(query: str, llm: BaseChatModel, k: int = 20) -> List[Document]:
    """HyDE를 사용하여 관련성 높은 문서를 검색합니다."""
    hyde_prompt = ChatPromptTemplate.from_template(
        "다음 질문에 대한 이상적인 답변(가상의 문서)을 한 단락으로 생성해주세요. "
        "이 답변은 벡터 검색에 사용될 것입니다.\n\n질문: {question}"
    )
    hyde_chain = hyde_prompt | llm

    print(f"[HyDE] 원본 질문: {query}")
    hypothetical_document = await hyde_chain.ainvoke({"question": query})
    hypothetical_content = hypothetical_document.content
    print(f"[HyDE] 생성된 가상 문서: {hypothetical_content[:100]}...")

    db = get_vector_db()
    if not db:
        return []
    return db.similarity_search(hypothetical_content, k=k)


async def advanced_rag_search(query: str, k: int, llm: BaseChatModel, reranker: HuggingFaceCrossEncoder) -> List[Document]:
    """HyDE와 Re-ranking을 결합한 고급 RAG 검색을 수행합니다."""
    print("[RAG] 고급 RAG 검색 시작...")

    initial_docs = await get_hyde_documents(query, llm, k=25)
    if not initial_docs:
        return []

    print(f"[RAG] HyDE로 {len(initial_docs)}개의 문서 검색 완료. Re-ranking 시작...")

    compressor = CrossEncoderReranker(model=reranker, top_n=k)

    reranked_docs = await asyncio.to_thread(
        compressor.compress_documents,
        documents=initial_docs,
        query=query
    )

    print(f"[RAG] Re-ranking 완료. 최종 {len(reranked_docs)}개 문서 선택.")
    return reranked_docs