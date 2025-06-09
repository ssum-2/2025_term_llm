import os
import asyncio
from typing import List
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# 글로벌 벡터DB 인스턴스
vector_db = None
# async
db_lock = asyncio.Lock()

def get_vector_db():
    global vector_db
    return vector_db


async def save_to_vector_db(docs: List[Document], batch_size: int=10):
    global vector_db

    if not docs:
        return
    embeddings = UpstageEmbeddings(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model="solar-embedding-1-large")
    # vector_db = FAISS.from_documents(docs, embeddings)

    # batch, 비동기 처리
    async with db_lock:
        # asyncio.to_thread를 사용해 이벤트 루프를 막지 않도록 설정
        loop = asyncio.get_running_loop()

        if vector_db is None:
            # db가 없으면 새로 생성
            await loop.run_in_executor(
                None,
                lambda: globals().update(vector_db=FAISS.from_documents(docs, embeddings))
            )
            print(f"[Embedding] 새로운 Vector DB 생성 및 {len(docs)}개 문서 저장 완료.")
        else:
            # db가 이미 있으면 문서 추가
            await loop.run_in_executor(
                None,
                lambda: vector_db.add_documents(docs)
            )
            print(f"[Embedding] 기존 Vector DB에 {len(docs)}개 문서 추가 완료.")

    return vector_db


def search_vector_db(query, k=5):
    db = get_vector_db()
    if db is None:
        return []
    return db.similarity_search(query, k=k)
