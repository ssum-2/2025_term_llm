import os
import re
from typing import List, Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from financial import FinancialTimeSeriesData

class Company(BaseModel):
    name: str = Field(..., description="증권사 이름. '알 수 없음'은 허용되지 않음.")
    opinion: str = Field(..., description="투자의견 (예: 매수, 중립, 매도, 강력매수, 보유 등)")
    target_price: int = Field(..., description="목표주가. 숫자만 포함해야 함.")
    rationale: str = Field(..., description="목표주가 및 투자의견에 대한 핵심 근거 요약 (1-2 문장)")
    source_file: str = Field(..., description="정보가 추출된 원본 파일명 (예: report.pdf)")
    source_page: str = Field(..., description="정보가 추출된 페이지 또는 청크 번호")

    @field_validator('target_price', mode='before')
    def clean_target_price(cls, v):
        if isinstance(v, str):
            nums = re.findall(r'\d+', v.replace(',', ''))
            if nums:
                return int("".join(nums))
        if isinstance(v, int):
            return v
        raise ValueError("목표주가에서 유효한 숫자를 추출할 수 없습니다.")

class FinalAnswer(BaseModel):
    summary: str = Field(..., description="모든 분석 자료를 종합하여 사용자의 질문에 대한 최종 답변.")
    sources: List[str] = Field(..., description="답변의 근거가 된 문서의 출처 리스트. '파일명 (p. 페이지번호)' 형식으로 기재. 예: ['삼성증권_리포트.pdf (p. 3)', '대신증권_리포트.pdf (p. 5)']")
    # `companies` 필드는 더 이상 최종 답변에 필요 없으므로 제거하거나 주석 처리합니다.
    # companies: List[Company] = Field(..., description="증권사별 분석 결과 리스트")
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
    cards: List[Company] = Field(..., description="텍스트에서 추출된 증권사별 분석 정보 리스트")

class ReportSummary(BaseModel):
    """PDF에서 추출한 증권사 리포트 요약 모델"""
    summary: str = Field(..., description="여러 증권사 리포트의 내용을 종합하여, 전반적인 시장 컨센서스와 핵심 논거를 요약합니다.")
    companies: List[Company] = Field(..., description="개별 증권사 분석 결과 리스트입니다.")

class NewsSummary(BaseModel):
    """뉴스 기사 요약 모델"""
    summary: str = Field(..., description="여러 뉴스 기사의 내용을 종합하여, 현재 시장 분위기와 주요 이슈를 2-3문장으로 요약합니다.")
    key_events: List[str] = Field(..., description="투자 결정에 영향을 미칠 수 있는 핵심 뉴스 이벤트나 토픽을 3가지 이내로 정리합니다.")

def build_extractor_chain(llm: BaseChatModel):
    """텍스트에서 구조화된 증권사 정보를 추출하는 Chain"""
    parser = PydanticOutputParser(pydantic_object=ExtractedInfo)
    prompt = PromptTemplate(
        template=(
            "당신은 금융 텍스트에서 구조화된 정보를 추출하는 AI입니다. "
            "주어진 텍스트에서 증권사 이름, 투자의견, 목표주가, 핵심 근거를 정확히 추출해주세요.\n"
            "여러 증권사의 정보가 있다면 모두 추출해야 합니다. 정보가 없다면 빈 리스트를 반환하세요.\n"
            "**[매우 중요한 규칙]** 만약 텍스트에서 명확한 '증권사 이름'(예: 삼성증권, BNK투자증권)을 찾을 수 없다면, "
            "절대 '종목명'(예: 삼성전자, SK하이닉스)을 증권사 이름으로 사용해서는 안됩니다. "
            "이 경우, 해당 정보는 추출하지 마세요.\n\n"
            "--- 텍스트 시작 ---\n"
            "{text}\n"
            "--- 텍스트 끝 ---\n\n"
            "추출 형식:\n{format_instructions}"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def build_report_summary_chain(llm: BaseChatModel):
    """[수정됨] 정제된 증권사 분석 데이터를 바탕으로 종합 요약을 생성하는 Chain"""
    parser = PydanticOutputParser(pydantic_object=ReportSummary)
    prompt = PromptTemplate(
        template="""당신은 증권사 리포트 분석 전문가입니다. 주어진 JSON 형식의 증권사별 분석 데이터를 바탕으로 종합적인 시장 컨센서스를 요약해야 합니다.

**[입력 데이터]**
`[리포트 내용]`에는 개별 증권사의 분석 결과가 JSON 객체 리스트 형식으로 제공됩니다. 각 객체는 `name`, `opinion`, `target_price` 필드를 포함합니다.

**[지시 사항]**
1. 모든 증권사의 투자의견과 목표주가를 종합하여, 전반적인 시장의 컨센서스(공통적인 의견, 목표주가 범위 등)와 핵심 투자 포인트를 `summary` 필드에 2-3 문장으로 요약해주세요.
2. `companies` 필드는 빈 리스트 `[]`로 반환해주세요. (이미 Python 코드에서 처리되었음)

**[출력 형식]**
{format_instructions}

**[리포트 내용 (JSON 리스트)]**
{context}
""",
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
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
1. 모든 뉴스 기사를 종합하여 현재 시장의 분위기나 주요 이슈를 `summary` 필드에 2-3문장으로 요약해주세요.
2. 주가에 영향을 미칠 수 있는 긍정적/부정적 핵심 이벤트나 주제를 `key_events` 리스트에 3개 이내로 정리해주세요.
3. **사용자의 질문과 무관하게**, 오직 제공된 `[뉴스 목록]`만을 기반으로 정보를 요약해야 합니다.

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