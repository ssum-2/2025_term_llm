import os
from functools import lru_cache
from langchain_upstage import ChatUpstage
from langchain_openai import ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from openai import OpenAI


# @lru_cache 데코레이터를 사용하여 함수 결과를 캐싱 -> 최초 호출 시 한 번만 모델을 로드
@lru_cache(maxsize=None)
def get_llm():
    """LLM 인스턴스를 반환합니다. 환경변수에 따라 Solar 또는 OpenAI 모델을 로드합니다."""
    mode = os.getenv("LLM_MODE", "solar")
    print(f"[Model Loader] Getting LLM for mode: {mode}")
    if mode == "solar":
        # ChatUpstage 모델은 특정 함수에서만 사용되므로 필요 시점에 맞게 분기
        return ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model_name="solar-pro-250422")
    else:
        return ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

@lru_cache(maxsize=None)
def get_mini_llm():
    """요약 등 가벼운 작업에 사용할 미니 LLM 인스턴스를 반환합니다."""
    mode = os.getenv("LLM_MODE", "solar")
    print(f"[Model Loader] Getting Mini LLM for mode: {mode}")
    if mode == "solar":
        return ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model_name="solar-1-mini-chat")
    else:
        # OpenAI API 키를 명시적으로 전달하도록 수정
        return ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")


@lru_cache(maxsize=None)
def get_reranker():
    """Reranker 모델 인스턴스를 반환합니다."""
    print("[Model Loader] Getting Reranker model")
    return HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

@lru_cache(maxsize=None)
def get_openai_client():
    """OpenAI 클라이언트 인스턴스를 반환합니다 (이미지 캡셔닝용)."""
    # 이 함수는 utils.py의 멀티모달 요약에 사용됩니다.
    print("[Model Loader] Getting OpenAI client")
    # 수정 전: if os.getenv("LLM_MODE", "solar") == "openai":
    if os.getenv("LLM_MODE", "solar") != "solar": # 수정 후
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return None