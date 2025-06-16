from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Optional, Dict
from pykrx import stock as pykrx_stock

class FinancialTimeSeriesData(BaseModel):
    ticker: str
    price: Dict[str, Optional[float]] = Field(default_factory=dict, description="주가 {YYYY-MM: 가격}")
    per: Dict[str, Optional[float]] = Field(default_factory=dict, description="PER {YYYY-MM: 값}")
    pbr: Dict[str, Optional[float]] = Field(default_factory=dict, description="PBR {YYYY-MM: 값}")
    dividend: Dict[str, Optional[float]] = Field(default_factory=dict, description="배당수익률 {YYYY-MM: %}")

def get_financial_timeseries_data_for_llm(ticker: str, years: int = 3) -> FinancialTimeSeriesData:
    """지정한 기간 동안의 월별 주가, PER, PBR, 배당수익률 데이터를 조회하고 Pydantic 모델로 반환합니다."""
    if not ticker:
        return FinancialTimeSeriesData(ticker="")

    try:
        today = datetime.now()
        start_date = today - timedelta(days=(years * 365)+30)
        today_str, start_date_str = today.strftime("%Y%m%d"), start_date.strftime("%Y%m%d")

        # time.sleep(0.2)
        price_df = pykrx_stock.get_market_ohlcv(start_date_str, today_str, ticker, freq='m')['종가']
        # time.sleep(0.2)
        fundamental_df = pykrx_stock.get_market_fundamental(start_date_str, today_str, ticker, freq='m')

        price_df.index = price_df.index.strftime('%Y-%m')
        fundamental_df.index = fundamental_df.index.strftime('%Y-%m')

        price_dict = price_df.to_dict()
        per_dict = fundamental_df['PER'].to_dict()
        pbr_dict = fundamental_df['PBR'].to_dict()
        dividend_dict = fundamental_df['DIV'].to_dict()

        return FinancialTimeSeriesData(
            ticker=ticker,
            price=price_dict,
            per=per_dict,
            pbr=pbr_dict,
            dividend=dividend_dict
        )
    except Exception as e:
        print(f"[ERROR] PyKRX 시계열 데이터 조회 실패 (Ticker: {ticker}): {e}")
        return FinancialTimeSeriesData(ticker=ticker)

def get_all_tickers() -> Dict[str, str]:
    """KRX 상장 종목 코드와 이름을 딕셔너리로 반환합니다."""
    print("[INFO] KRX 전체 종목 티커 조회 중...")
    try:
        tickers = pykrx_stock.get_market_ticker_list(market="ALL")
        return {
            pykrx_stock.get_market_ticker_name(ticker): ticker
            for ticker in tickers
            if pykrx_stock.get_market_ticker_name(ticker) is not None
        }
    except Exception as e:
        print(f"[ERROR] KRX 티커 목록 조회 실패: {e}")
        return {}