import re
from typing import List, Dict, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- 데이터 모델 ---
class ExtractedAssertion(BaseModel):
    """텍스트에서 추출된 원시 '주장' 데이터"""
    subject: str = Field(..., description="이 주장이 다루고 있는 핵심 주제 (예: 'SK하이닉스 4분기 실적')")
    assertion: str = Field(..., description="텍스트의 핵심 주장 또는 결론 요약")
    evidence: str = Field(..., description="위 주장을 뒷받침하는 핵심 근거 또는 데이터")
    sentiment: str = Field(..., description="주제에 대한 주장의 뉘앙스 ('긍정적', '부정적', '중립')")
    source_entity: str = Field("작성자", description="이 주장을 펼치는 주체. 명시된 기관/회사명이 없으면 '작성자'라고 표기.")

class ExtractedInfo(BaseModel):
    assertions: List[ExtractedAssertion]

class KeyClaim(BaseModel):
    """개별 주장의 핵심 내용"""
    claim: str = Field(..., description="핵심 주장 또는 결론 요약")
    evidence: str = Field(..., description="주장을 뒷받침하는 핵심 근거 또는 데이터")
    sentiment: str = Field(..., description="주장에 대한 뉘앙스 ('긍정적', '부정적', '중립')")

class EntityAnalysis(BaseModel):
    """'하나의 분석 주체'에 대한 요약 정보를 담는 모델"""
    entity_name: str = Field(..., description="분석 주체의 이름)")
    main_stance: str = Field(..., description="이 주체가 제시하는 여러 주장을 종합한 핵심 입장 요약 (5-7 문장)")
    key_claims: List[KeyClaim] = Field(..., description="이 주체가 제시하는 주요 주장 목록 (최대 5개)")

class ReportSummary(BaseModel):
    """UI에 표시될 최종 PDF 분석 결과"""
    overall_insight: str = Field(..., description="모든 분석 주체들의 의견을 종합한 최종 인사이트 (5-7 문장)")
    entity_analyses: List[EntityAnalysis] = Field(..., description="분석 주체별 요약 정보 목록")

class NewsSummary(BaseModel):
    """뉴스 기사 요약 모델"""
    summary: str = Field(..., description="뉴스 기사들의 핵심 내용을 종합한 요약")
    key_events: List[str] = Field(..., description="주가에 영향을 미칠 수 있는 긍정적/부정적 핵심 이벤트 목록")
    articles: List[Dict[str, Any]] = Field(..., description="요약의 기반이 된 원본 뉴스 기사 목록 (제목, URL 등 포함)")


class GrandSummary(BaseModel):
    """PDF 리포트와 최신 뉴스를 종합한 최종 요약"""
    title: str = Field(..., description="종합 요약의 제목 (예: '삼성전자 투자 분석 종합')")
    content: str = Field(..., description="PDF와 뉴스 정보를 종합하여 생성한 최종 분석 내용 (5-7 문장)")

class FinalAnswer(BaseModel):
    """사용자 질문에 대한 최종 답변 모델"""
    summary: str
    sources: List[str]
    news_summary: Optional[str]
    suggested_questions: Optional[List[str]]


def build_extractor_chain(llm: BaseChatModel):
    """텍스트에서 구조화된 '주장'을 추출하는 체인"""
    parser = PydanticOutputParser(pydantic_object=ExtractedInfo)
    prompt = PromptTemplate(
        template=(
            "당신은 편견 없는 최고의 금융 분석가입니다. 주어진 텍스트는 '{source_entity_name}'에서 작성한 리포트의 일부입니다."
            "이 텍스트에서 객관적인 '핵심 주장(Assertion)'과 그 근거를 추출해야 합니다.\n"
            "**[추출 지시사항]**\n"
            "주어진 텍스트에서 논리적 완결성을 가진 모든 '주장' 단위를 찾아내고, 각 주장에 대해 다음 항목을 추출하여 목록으로 만드세요:\n"
            "1. `subject`: 이 주장이 무엇에 대해 이야기하고 있는지 명확한 '주제'\n"
            "2. `assertion`: 그래서 결론이 무엇인지, 간결하게 요약된 '주장'\n"

            "3. `evidence`: 그 주장을 뒷받침하는 '근거'\n"
            "4. `sentiment`: 해당 '주제'에 대해 '주장'이 가지는 뉘앙스 ('긍정적', '부정적', '중립')\n"
            "5. `source_entity`: 이 주장의 출처는 미리 제공된 '{source_entity_name}' 입니다. 이 값을 그대로 사용하세요.\n"
            "--- 분석 대상 텍스트 시작 ---\n{text}\n--- 분석 대상 텍스트 끝 ---\n"
            "결과는 다음 포맷에 맞춰 출력해주세요:\n{format_instructions}"
        ),
        input_variables=["text", "source_entity_name"], # source_entity_name 추가
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def build_entity_summary_chain(llm: BaseChatModel):
    """한 주체의 여러 주장을 받아서 하나의 EntityAnalysis로 요약하는 체인"""
    parser = PydanticOutputParser(pydantic_object=EntityAnalysis)
    prompt = PromptTemplate(
        template="""당신은 수석 애널리스트입니다. 특정 분석 주체(`entity_name`)가 제시한 여러 주장들이 아래에 목록으로 제공됩니다.
    이 주장들을 종합하여 이 주체의 핵심 입장과 주요 근거를 요약해주세요.

    **[분석 대상]**
    - 분석 주체 이름: {entity_name}
    - 이 주체가 제시한 주장 목록 (주장, 근거, 뉘앙스):
    {claims_context}

    **[지시 사항]**
    1. `entity_name`을 그대로 사용하여 필드를 채워주세요.
    2. 제시된 모든 주장을 종합하여, 이 주체의 **핵심 입장(main_stance)**을 3~5 문장으로 요약해주세요.
    3. 가장 중요하다고 생각되는 **핵심 주장(key_claims)을 최대 3개**까지 간추려 리스트를 완성해주세요. 각 주장의 논리(주장, 근거, 뉘앙스)를 명확히 담아야 합니다.

    **[출력 형식]**
    {format_instructions}
    """,
        input_variables=["entity_name", "claims_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def build_news_summary_chain(llm: BaseChatModel):
    """뉴스 정보만으로 종합 요약을 생성하는 체인"""
    class NewsSummaryForGeneration(BaseModel):
        summary: str
        key_events: List[str]

    parser = PydanticOutputParser(pydantic_object=NewsSummaryForGeneration)
    prompt = PromptTemplate(
        template="""당신은 시장 동향 분석가입니다. 주어진 최신 뉴스 기사 목록을 바탕으로 시장의 주요 동향과 이벤트를 요약해야 합니다.\n
        **[입력 데이터]**\n`[뉴스 목록]`에는 여러 뉴스 기사의 제목과 내용 일부가 제공됩니다.\n
        **[지시 사항]**\n1. 모든 뉴스 기사를 종합하여 현재 시장의 분위기나 주요 이슈를 `summary` 필드에 3-5문장으로 요약해주세요.\n2. 주가에 영향을 미칠 수 있는 긍정적/부정적 핵심 이벤트나 주제를 `key_events` 리스트에 5개 이내로 정리해주세요.\n
        **[출력 형식]**\n{format_instructions}\n\n**[뉴스 목록]**\n{news_articles}""",
        input_variables=["news_articles"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def build_grand_summary_chain(llm: BaseChatModel):
    """PDF 리포트 요약과 뉴스 요약을 종합하여 최종 인사이트를 생성하는 체인"""
    parser = PydanticOutputParser(pydantic_object=GrandSummary)
    prompt = PromptTemplate(
        template="""당신은 최고의 투자 전략가입니다. 아래 제공된 두 가지 핵심 정보, 즉 '증권사 리포트 종합 분석'과 '최신 뉴스 동향'을 모두 고려하여,
투자자를 위한 최종적이고 종합적인 분석을 제공해야 합니다.

**[핵심 정보 1: 증권사 리포트 종합 분석]**
{report_insight}

**[핵심 정보 2: 최신 뉴스 동향 요약]**
{news_summary}

**[지시 사항]**
1.  두 정보를 통합하여 현재 상황에 대한 균형 잡힌 시각을 제시해주세요. 리포트의 깊이 있는 분석과 뉴스의 시의성을 결합해야 합니다.
2.  분석의 핵심 내용을 담은 `title`을 작성해주세요. (예: 'OO기업, 기회와 위협 요인 종합 분석')
3.  리포트와 뉴스를 종합한 최종 결론을 `content`에 3~5문장의 명확하고 간결한 문단으로 작성해주세요.

**[출력 형식]**
{format_instructions}
""",
        input_variables=["report_insight", "news_summary"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def build_final_answer_chain(llm: BaseChatModel):
    """사용자의 특정 질문에 답변하고, 답변의 근거가 된 출처를 명시하는 최종 Chain"""
    parser = PydanticOutputParser(pydantic_object=FinalAnswer)
    prompt = PromptTemplate(
        template="""당신은 최고의 펀드매니저입니다. 사용자의 질문에 답하기 위해 주어진 모든 정보를 종합적으로 분석해야 합니다.

        **[분석 자료]**
        1.  **참고 문서 (RAG 검색 결과)**: `[참고 문서]`에는 여러 문서 조각들이 제공됩니다. 각 문서 조각의 시작 부분에는 `(출처: 파일명, 페이지: 페이지번호)`와 같은 메타데이터가 있습니다.
        2.  **재무 데이터**: `[재무 데이터]`는 분석 대상 기업의 시계열 재무 지표입니다.
        3.  **최신 뉴스 요약**: `[뉴스 요약]`은 시장의 최근 동향을 요약한 정보입니다.

        **[매우 중요한 지시 사항]**
        1. `[사용자 질문]`의 의도를 명확히 파악하고, `[분석 자료]`를 종합하여 질문에 대한 명확하고 논리적인 답변을 `summary`에 작성해주세요.
        2. 답변을 작성할 때 **어떤 문서를 참고했는지 반드시 기억**해야 합니다.
        3. 답변의 근거가 된 문서들의 출처 정보(파일명과 페이지 번호)를 `sources` 리스트에 정확하게 추가해주세요. **이것은 매우 중요합니다.**
        4. 만약 사용자의 질문이 분석과 무관하다면, `summary`에 "요청하신 질문은 제공된 정보로 답변하기 어렵습니다."라고 명시하고, `suggested_questions`에 적절한 대체 질문을 제안하세요. 이 경우 `sources`는 빈 리스트로 반환합니다.

        **[출력 형식]**\n{format_instructions}\n
        **[사용자 질문]**\n{question}\n
        **[참고 문서 (RAG 검색 결과)]**\n{context}\n
        **[재무 데이터]**\n{financial}\n
        **[최신 뉴스 요약]**\n{news_summary}""",
        input_variables=["question", "context", "financial", "news_summary"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser