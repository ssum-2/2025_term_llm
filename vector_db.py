import os
import shutil
import json
import asyncio
from typing import List
from langchain_upstage import UpstageEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import hashlib


# --- 캐시 경로 설정 ---
CACHE_DIR = "./.cache/vector_db"
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")
EMBEDDED_HASHES_PATH = os.path.join(CACHE_DIR, "embedded_hashes.json")

# 글로벌 벡터DB 인스턴스
vector_db: FAISS | None = None
# async lock
db_lock = asyncio.Lock()
# 임베딩된 문서 해시 기록
embedded_hashes: set[str] = set()


def _get_embedding_model():
    """LLM_MODE 환경변수에 따라 적절한 임베딩 모델 인스턴스를 반환합니다."""
    mode = os.getenv("LLM_MODE", "solar")
    if mode == "solar":
        print("[Embedding] Using Upstage Embeddings")
        return UpstageEmbeddings(model="solar-embedding-1-large")
    else: # 'openai' 또는 'gpt' 모드
        print("[Embedding] Using OpenAI Embeddings")
        return OpenAIEmbeddings(model="text-embedding-3-small")


def _hash_doc(doc: Document) -> str:
    """Document 객체의 내용과 메타데이터를 기반으로 고유한 해시를 생성합니다."""
    # 내용 + 파일명 + 페이지/청크 정보를 합쳐서 해시
    combined_string = f"{doc.page_content}|{doc.metadata.get('source_file')}|{doc.metadata.get('page')}"
    return hashlib.md5(combined_string.encode()).hexdigest()

def load_vector_db_from_disk():
    """서버 시작 시 디스크에서 벡터 DB와 해시 기록을 로드합니다."""
    global vector_db, embedded_hashes
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDED_HASHES_PATH):
        try:
            print("[Cache] 디스크에서 Vector DB 로딩 중...")
            embeddings = _get_embedding_model()
            vector_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

            with open(EMBEDDED_HASHES_PATH, 'r') as f:
                embedded_hashes = set(json.load(f))
            print(f"[Cache] 로딩 완료. {len(embedded_hashes)}개의 문서가 DB에 존재합니다.")
        except Exception as e:
            print(f"[Cache ERROR] 디스크에서 DB 로딩 실패: {e}. DB를 새로 시작합니다.")
            vector_db = None
            embedded_hashes = set()
    else:
        print("[Cache] 저장된 Vector DB가 없습니다. 새로 시작합니다.")

def get_vector_db():
    global vector_db
    return vector_db

def clear_vector_db():
    """메모리와 디스크에 있는 벡터 DB와 해시 기록을 초기화합니다."""
    global vector_db, embedded_hashes
    vector_db = None
    embedded_hashes.clear()
    if os.path.exists(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
            print("[INFO] 디스크의 Vector DB 캐시가 삭제되었습니다.")
        except Exception as e:
            print(f"[ERROR] 디스크 캐시 삭제 실패: {e}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("[INFO] Vector DB와 임베딩 기록이 초기화되었습니다.")

async def save_to_vector_db(docs: List[Document]):
    global vector_db, embedded_hashes

    if not docs:
        return

    # 해시를 기준으로 아직 임베딩되지 않은 고유한 문서만 필터링
    unique_docs = [d for d in docs if _hash_doc(d) not in embedded_hashes]
    if not unique_docs:
        print("[Embedding] 추가할 새로운 문서가 없습니다. (모두 중복)")
        return

    print(f"[Embedding] 총 {len(docs)}개 문서 중 {len(unique_docs)}개의 새로운 문서를 임베딩합니다.")
    embeddings = _get_embedding_model()

    async with db_lock:
        loop = asyncio.get_running_loop()
        if vector_db is None:
            new_db = await loop.run_in_executor(None, FAISS.from_documents, unique_docs, embeddings)
            globals()['vector_db'] = new_db
            print(f"[Embedding] 새로운 Vector DB 생성 및 {len(unique_docs)}개 문서 저장 완료.")
        else:
            # FAISS는 비동기 add_documents를 지원하지 않으므로 run_in_executor 사용
            await loop.run_in_executor(None, vector_db.add_documents, unique_docs)
            print(f"[Embedding] 기존 Vector DB에 {len(unique_docs)}개 문서 추가 완료.")

        # 임베딩된 문서의 해시를 기록에 추가
        for d in unique_docs:
            embedded_hashes.add(_hash_doc(d))

        # 변경 사항을 디스크에 저장
        try:
            await loop.run_in_executor(None, vector_db.save_local, FAISS_INDEX_PATH)
            with open(EMBEDDED_HASHES_PATH, 'w') as f:
                json.dump(list(embedded_hashes), f)
            print("[Cache] Vector DB와 해시 기록을 디스크에 저장했습니다.")
        except Exception as e:
            print(f"[Cache ERROR] 디스크 저장 실패: {e}")

def search_vector_db(query: str, k: int = 10, filter_criteria: dict | None = None) -> List[Document]:
    db = get_vector_db()
    if db is None:
        print("[Warning] Vector DB가 비어있습니다. 검색을 수행할 수 없습니다.")
        return []

    if not query:  # 쿼리가 비어있는 경우 에러 방지
        print("[Warning] 검색어가 비어있어, 의미 없는 검색을 방지합니다.")
        return []

    if filter_criteria:
        return db.similarity_search(query, k=k, filter=filter_criteria)
    else:
        return db.similarity_search(query, k=k)


def search_by_metadata(filter_criteria: dict, k: int = 100) -> List[Document]:
    """쿼리 없이 메타데이터만으로 문서를 검색합니다."""
    db = get_vector_db()
    if db is None:
        print("[Warning] Vector DB가 비어있습니다. 검색을 수행할 수 없습니다.")
        return []

    # FAISS의 내부 docstore에서 직접 메타데이터를 확인하여 필터링
    matching_docs = []
    if hasattr(db, 'docstore') and hasattr(db, 'index_to_docstore_id'):
      for doc_id in db.index_to_docstore_id.values():
          doc = db.docstore.search(doc_id)
          if doc and all(item in doc.metadata.items() for item in filter_criteria.items()):
              matching_docs.append(doc)
          if len(matching_docs) >= k:
              break
    return matching_docs