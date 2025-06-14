import os
from gnews import GNews
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any

def get_latest_news(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """GNews를 사용하여 특정 쿼리(종목명)에 대한 최신 뉴스를 가져옵니다."""
    print(f"[News] '{query}' 관련 최신 뉴스 검색 중...")
    try:
        google_news = GNews(language='ko', country='KR', period='7d')
        news = google_news.get_news(query)

        unique_news = []
        seen_titles = set()
        for item in news[:max_results]:
            title = item.get('title')
            if title and title not in seen_titles:
                unique_news.append({
                    "title": title,
                    "url": item.get('url'),
                    "published_date": item.get('published date'),
                    "publisher": item.get('publisher', {}).get('title')
                })
                seen_titles.add(title)
        print(f"[News] {len(unique_news)}개의 고유한 뉴스 기사 발견.")
        return unique_news
    except Exception as e:
        print(f"[ERROR] 뉴스 검색 실패: {e}")
        return []

async def summarize_news_async(articles: List[Dict[str, Any]], llm: BaseChatModel) -> str:
    """여러 뉴스 기사 제목을 바탕으로 동향을 비동기적으로 요약합니다."""
    if not articles:
        return "최신 뉴스가 없습니다."

    titles = "\n".join([f"- {a['title']}" for a in articles])

    prompt = ChatPromptTemplate.from_template(
        "다음은 특정 종목에 대한 최신 뉴스 기사 제목 목록입니다. "
        "이 뉴스들을 종합하여 현재 시장의 분위기나 주요 이슈를 2~3문장으로 요약해주세요.\n\n"
        "**뉴스 목록:**\n{news_titles}\n\n"
        "**종합 요약:**"
    )

    chain = prompt | llm

    try:
        summary = await chain.ainvoke({"news_titles": titles})
        return summary.content
    except Exception as e:
        print(f"[ERROR] 뉴스 요약 실패: {e}")
        return "뉴스 요약 중 오류가 발생했습니다."