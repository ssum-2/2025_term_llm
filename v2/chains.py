import os, re
from typing import List, Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field



class KeyAssertion(BaseModel):
    subject: str = Field(..., description="이 주장이 다루고 있는 핵심 주제")
    assertion: str = Field(..., description="텍스트의 핵심 주장 또는 결론 요약")
    evidence: str = Field(..., description="위 주장을 뒷받침하는 핵심 근거 또는 데이터")
    sentiment: str = Field(..., description="해당 '주제'에 대한 주장의 뉘앙스 ('긍정적', '부정적', '중립')")
    source_entity: str = Field("작성자", description="이 주장을 펼치는 주체. 명시되지 않은 경우 '작성자'로 표기.")
    source_file: str = Field(..., description="정보가 추출된 원본 파일명")
    source_page: str = Field(..., description="정보가 추출된 페이지 번호")

class FinalAnswer(BaseModel):
    """사용자 질문에 대한 최종 답변 모델"""
    summary: str = Field(..., description="모든 분석 자료를 종합하여 사용자의 질문에 대한 최종 답변.")
    sources: List[str] = Field(..., description="답변의 근거가 된 문서의 출처 리스트. '파일명 (p. 페이지번호)' 형식으로 기재.")
    news_summary: Optional[str] = Field(None, description="질문과 관련된 최신 뉴스 동향 요약")
    suggested_questions: Optional[List[str]] = Field(None, description="사용자가 엉뚱한 질문을 했을 경우, 대신 할만한 추천 질문 3가지")

class PreInsight(BaseModel):
    highlights: List[str] = Field(..., description="리포트와 재무 데이터를 종합하여 투자자가 가장 주목해야 할 핵심적인 인사이트 3~4가지. 긍정적/부정적 요소를 모두 포함해야 함.")
    suggested_questions: List[str] = Field(..., description="분석된 내용을 바탕으로 사용자가 궁금해할 만한 후속 질문 3가지.")

class GlobalOverview(BaseModel):
    overview: str = Field(..., description="PDF 전부를 8~10줄로 요약")
    key_risks: List[str] = Field(..., description="최대 3가지 핵심 리스크")
    key_triggers: List[str] = Field(..., description="주가 상승/하락 촉발 요인 3가지")

class ExtractedInfo(BaseModel):
    assertions: List[KeyAssertion] = Field(..., description="텍스트에서 추출된 핵심 주장(KeyAssertion) 목록")

class ReportSummary(BaseModel):
    """PDF에서 추출한 정보를 종합 요약하는 모델"""
    overall_insight: str = Field(..., description="모든 주장을 종합하여 분석 대상에 대한 핵심 인사이트를 3-4 문장으로 요약합니다.")
    assertions_by_theme: Dict[str, List[KeyAssertion]] = Field(..., description="핵심 '주제(subject)'별로 관련된 주장(KeyAssertion)들을 묶은 딕셔너리")

class NewsSummary(BaseModel):
    """뉴스 기사 요약 모델"""
    summary: str = Field(..., description="여러 뉴스 기사의 내용을 종합하여, 현재 시장 분위기와 주요 이슈를 2-3문장으로 요약합니다.")
    key_events: List[str] = Field(..., description="투자 결정에 영향을 미칠 수 있는 핵심 뉴스 이벤트나 토픽을 3가지 이내로 정리합니다.")

def build_extractor_chain(llm: BaseChatModel):
    """ 모든 종류의 금융/경제 문서에서 '핵심 주장(KeyAssertion)'을 추출하는 범용 Chain"""
    parser = PydanticOutputParser(pydantic_object=ExtractedInfo)
    prompt = PromptTemplate(
        template=(
            "당신은 편견 없는 최고의 금융 분석가입니다. 당신의 임무는 주어진 텍스트의 종류(증권사 리포트, 경제 뉴스, 연구소 보고서 등)에 상관없이, 그 안에 담긴 객관적인 '핵심 주장(Key Assertion)'과 그 근거를 추출하는 것입니다.\n\n"
            "**[추출 지시사항]**\n"
            "주어진 텍스트에서 논리적 완결성을 가진 모든 '주장' 단위를 찾아내고, 각 주장에 대해 다음 항목을 추출하여 목록으로 만드세요:\n"
            "1. `subject`: 이 주장이 무엇에 대해 이야기하고 있는지 명확한 '주제'\n"
            "2. `assertion`: 그래서 결론이 무엇인지, 간결하게 요약된 '주장'\n"
            "3. `evidence`: 그 주장을 뒷받침하는 '근거' (데이터, 현상, 논리 등)\n"
            "4. `sentiment`: 해당 '주제'에 대해 '주장'이 가지는 뉘앙스 ('긍정적', '부정적', '중립')\n"
            "5. `source_entity`: 이 주장을 펼치는 주체. 명시된 기관/회사명이 있으면 쓰고, 없으면 '작성자'라고 표기하세요.\n\n"
            "**[매우 중요한 원칙]**\n"
            "- 텍스트에 여러 주장이 있다면, 모두 별개의 항목으로 추출해야 합니다.\n"
            "- 추출할 정보가 전혀 없다면 빈 리스트를 반환하세요.\n\n"
            "--- 분석 대상 텍스트 시작 ---\n"
            "{text}\n"
            "--- 분석 대상 텍스트 끝 ---\n\n"
            "결과는 다음 포맷에 맞춰 출력해주세요:\n{format_instructions}"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def build_report_summary_chain(llm: BaseChatModel):
    """추출된 모든 '핵심 주장'들을 종합하여 주제별로 정리하고, 통찰력 있는 요약을 생성하는 Chain"""
    parser = PydanticOutputParser(pydantic_object=ReportSummary)
    prompt = PromptTemplate(
        template="""당신은 수석 애널리스트입니다. 여러 출처에서 나온 분석 내용들이 '주제'별로 정리되어 아래에 제공됩니다.
    이것들을 종합하여 시장과 종목에 대한 깊이 있는 '종합 인사이트'를 도출해주세요.

    **[분석 데이터 (주제별 주장 목록)]**
    {context}

    **[지시 사항]**
    1.  제공된 모든 정보를 바탕으로, 투자자가 가장 중요하게 생각해야 할 **종합적인 인사이트(overall_insight)**를 최대 10 문장으로 작성해주세요. 여기에는 긍정적 측면, 부정적 측면, 그리고 향후 전망이 균형 있게 포함되어야 합니다.
    2.  `assertions_by_theme` 필드는 입력받은 데이터를 그대로 복사하지 말고, **반드시 빈 객체 `{{}}`** 로 반환해주세요. 이 필드는 Python 코드에서 직접 채울 것입니다.

    **[출력 형식]**
    {format_instructions}
    """,
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions().replace('{}', '{{}}')},
    )
    return prompt | llm | parser


def build_news_summary_chain(llm: BaseChatModel):
    """[신규] 뉴스 정보만으로 종합 요약을 생성하는 Chain"""
    parser = PydanticOutputParser(pydantic_object=NewsSummary)
    prompt = PromptTemplate(
        template="""당신은 시장 동향 분석가입니다. 주어진 최신 뉴스 기사 목록을 바탕으로 시장의 주요 동향과 이벤트를 요약해야 합니다.

    **[입력 데이터]**
    `[뉴스 목록]`에는 여러 뉴스 기사의 제목과 내용 일부가 제공됩니다.

    **[지시 사항]**
    1. 모든 뉴스 기사를 종합하여 현재 시장의 분위기나 주요 이슈를 `summary` 필드에 3-5문장으로 요약해주세요.
    2. 주가에 영향을 미칠 수 있는 긍정적/부정적 핵심 이벤트나 주제를 `key_events` 리스트에 3개 이내로 정리해주세요.

    **[출력 형식]**
    {format_instructions}

    **[뉴스 목록]**
    {news_articles}
    """,
        input_variables=["news_articles"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser


def build_final_answer_chain(llm: BaseChatModel):
    """[수정됨] 사용자의 특정 질문에 답변하고, 답변의 근거가 된 출처를 명시하는 최종 Chain"""
    parser = PydanticOutputParser(pydantic_object=FinalAnswer)
    prompt = PromptTemplate(
        template="""당신은 최고의 펀드매니저입니다. 사용자의 질문에 답하기 위해 주어진 모든 정보를 종합적으로 분석해야 합니다.

    **[분석 자료]**
    1.  **참고 문서 (RAG 검색 결과)**: `[참고 문서]`에는 여러 문서 조각들이 제공됩니다. 각 문서 조각의 시작 부분에는 `(출처: 파일명, 페이지: 페이지번호)`와 같은 메타데이터가 있습니다.
    2.  **재무 데이터**: `[재무 데이터]`는 분석 대상 기업의 시계열 재무 지표입니다.
    3.  **최신 뉴스 요약**: `[뉴스 요약]`은 시장의 최근 동향을 요약한 정보입니다.

    **[매우 중요한 지시 사항]**
    1.  `[사용자 질문]`의 의도를 명확히 파악하고, `[분석 자료]`를 종합하여 질문에 대한 명확하고 논리적인 답변을 `summary`에 작성해주세요.
    2.  답변을 작성할 때 **어떤 문서를 참고했는지 반드시 기억**해야 합니다.
    3.  답변의 근거가 된 문서들의 출처 정보(파일명과 페이지 번호)를 `sources` 리스트에 정확하게 추가해주세요. **이것은 매우 중요합니다.**
    4.  만약 사용자의 질문이 분석과 무관하다면, `summary`에 "요청하신 질문은 제공된 정보로 답변하기 어렵습니다."라고 명시하고, `suggested_questions`에 적절한 대체 질문을 제안하세요. 이 경우 `sources`는 빈 리스트로 반환합니다.

    **[출력 형식]**
    {format_instructions}

    **[사용자 질문]**
    {question}

    **[참고 문서 (RAG 검색 결과)]**
    {context}

    **[재무 데이터]**
    {financial}

    **[최신 뉴스 요약]**
    {news_summary}
    """,
        input_variables=["question", "context", "financial", "news_summary"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def build_preinsight_chain(llm: BaseChatModel):
    # ... (기존 코드와 동일)
    parser = PydanticOutputParser(pydantic_object=PreInsight)
    prompt = PromptTemplate(
        template="""당신은 뛰어난 애널리스트입니다. 방금 업로드된 증권 리포트와 관련 재무 지표를 검토하고, 투자자가 가장 먼저 알아야 할 핵심 내용과 궁금해할 만한 질문을 뽑아내야 합니다.

    **[분석 자료]**

    **1. 문서 요약 (증권사 코멘트):**
    {context}

    **2. 재무 지표:**
    {financial}

    **3. 최근 1주일 뉴스 헤드라인:**
    {news_titles}

    **[지시사항]**

    1.  `[분석 자료]`를 종합하여, 이 종목의 투자 매력도(긍정/부정 요인 포함)를 가장 잘 보여주는 **핵심 하이라이트 3~4개**를 도출하세요.
    2.  재무지표 추세(모멘텀·밸류에이션)와 증권사 의견을 교차검증해 **투자자 관점에서 반드시 확인해야 할 후속 질문 3개**를 제안하세요.
    3.  결과를 아래 JSON 형식으로 반환하세요.

    {format_instructions}
    """,
        input_variables=["context", "financial", "news_titles"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def build_global_overview_chain(llm: BaseChatModel):
    # ... (기존 코드와 동일)
    parser = PydanticOutputParser(pydantic_object=GlobalOverview)
    prompt = PromptTemplate(
        template=(
            "다음은 여러 PDF 리포트에서 추출한 **요약문 집합**입니다.\n"
            "이를 종합하여 종목 전반의 투자가치를 ①10줄 내외 ‘개요’, "
            "②핵심 리스크 ③주요 트리거 로 나눠 주세요.\n\n"
            "{format_instructions}\n\n"
            "--- 요약문 집합 시작 ---\n"
            "{context}\n"
            "--- 요약문 집합 끝 ---"
        ),
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser