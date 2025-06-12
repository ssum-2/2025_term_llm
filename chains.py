from langchain_upstage import ChatUpstage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import os


class Company(BaseModel):
    name: str = Field(..., description="증권사 이름")
    opinion: str = Field(..., description="의견 (매수/중립/매도 등)")
    target_price: Union[str, int] = Field(..., description="목표주가")
    reason: str = Field(..., description="근거 요약")
    source: str = Field(..., description="출처 (리포트명, 페이지)")

class FinalAnswer(BaseModel):
    summary: str = Field(..., description="여러 리포트를 종합한 전체 요약")
    companies: List[Company] = Field(..., description="증권사별 분석 결과 리스트")
    financial_data: Dict = Field(..., description="요청된 종목의 재무 데이터")

class FinalAnswer_v2(BaseModel):
    summary: str = Field(..., description="여러 리포트를 종합한 전체 요약")
    companies: List[Company] = Field(..., description="증권사별 분석 결과 리스트")
    financial_TimeSeries: Dict[str, float] = Field(..., description="요청된 종목의 재무 데이터")



def get_report_chain():
    mode = os.getenv("LLM_MODE", "solar")
    if mode == "solar":
        llm = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model_name="solar-pro-250422")
    else:
        llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    parser = PydanticOutputParser(pydantic_object=FinalAnswer_v2)

    # [TEST] output이 원하는 모양으로 나올 때까지 prompt를 수정할 것
    # format instruction 주입
    prompt = PromptTemplate(
        template="""당신은 금융 애널리스트입니다. 주어진 여러 개의 문서 조각과 재무 데이터를 바탕으로 사용자의 질문에 답변해야 합니다.
        반드시 아래 지시사항과 출력 형식에 맞춰 답변을 생성해주세요.
    
        [지시사항]
        1.  아래 [참고 문서]에는 여러 출처의 내용이 '--- 문서 출처: [파일명], 페이지: [페이지] ---' 형식으로 구분되어 있습니다.
        2.  **가장 중요한 규칙: 각 '파일명'을 기준으로 모든 관련 정보를 종합하여, 증권사별로 단 하나의 카드만 생성해야 합니다.**
        3.  **절대 동일한 증권사 이름(name)으로 여러 개의 객체를 만들지 마세요.** 예를 들어, '대신증권'에서 여러 정보가 발견되면, 그 정보들을 모두 요약하여 '대신증권' 객체 단 하나에 포함시키세요.
        4.  모든 정보를 종합하여 최종 요약을 작성하고, 요청된 JSON 형식으로만 답변하세요.
    
        [출력 형식]
        {format_instructions}
    
        [사용자 질문]:
        {question}
    
        [참고 문서]:
        {context}
    
        [재무 데이터]:
        {financial}
        """,
        input_variables=["question", "context", "financial"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    return chain

