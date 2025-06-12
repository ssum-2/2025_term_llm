from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pandas as pd
from typing import Optional, Dict, Any
import yfinance as yf
from pykrx import stock as pykrx_stock

class FinancialData(BaseModel):
    ticker: str = Field(..., description="종목 코드")
    price: Optional[float] = Field(None, description="주가")
    per: Optional[float] = Field(None, description="PER")
    pbr: Optional[float] = Field(None, description="PBR")
    dividend: Optional[float] = Field(None, description="배당수익률")
class Financial_TimeSeries_Data(BaseModel):
    ticker: str = Field(..., description="종목 코드")
    price: Optional[Dict[str, float]] = Field(None, description="주가 {YYYY-MM: 가격}")
    per: Optional[Dict[str, float]] = Field(None, description="PER  {YYYY-MM: 값}")
    pbr: Optional[Dict[str, float]] = Field(None, description="PBR  {YYYY-MM: 값}")
    dividend: Optional[Dict[str, float]] = Field(None, description="배당수익률 {YYYY-MM: %}")

def get_pykrx_data(ticker: str, indicator: str) -> Dict[str, Any]:
    try:
        today = datetime.now().strftime("%Y%m%d")
        if indicator == "price":
            df = pykrx_stock.get_market_ohlcv(today, today, ticker)
            return {"ticker": ticker, "price": float(df['종가'].iloc[-1])}
        elif indicator == "per":
            df = pykrx_stock.get_market_fundamental(today, today, ticker)
            return {"ticker": ticker, "per": float(df['PER'].iloc[-1])}
        elif indicator == "pbr":
            df = pykrx_stock.get_market_fundamental(today, today, ticker)
            return {"ticker": ticker, "pbr": float(df['PBR'].iloc[-1])}
        else:
            return {"error": "지원하지 않는 지표입니다"}
    except Exception as e:
        return {"error": f"PyKRX 데이터 가져오기 실패: {str(e)}"}
def get_pykrx_data_nY(ticker: str, nY: int=3) -> Dict[str, Any]:
    try:
        today = datetime.now().strftime("%Y%m%d")
        nY_ago =  (datetime.now() - timedelta(days=nY*(365+30))).strftime("%Y%m%d")
        p_df = pykrx_stock.get_market_ohlcv(nY_ago, today, ticker, freq='m')['종가']
        p_df = p_df.rename(index={p_df.index[-1]: pd.to_datetime(today)})
        p_df = p_df.rename(index=lambda x: x.strftime("%Y-%m"))
        p_dict = p_df.to_dict()
        f_df = pykrx_stock.get_market_fundamental(nY_ago, today, ticker, freq='m')[['PER', 'PBR', 'DIV']]
        f_df = f_df.rename(index={f_df.index[-1]: pd.to_datetime(today)})
        f_df = f_df.rename(index=lambda x: x.strftime("%Y-%m"))
        f_dict = f_df.to_dict()

        return {"ticker": ticker, "price": p_dict, "per": f_dict['PER'], "pbr": f_dict['PBR'], "dividend": f_dict['DIV']}
    except Exception as e:
        return {"error": f"PyKRX 데이터 가져오기 실패: {str(e)}"}

def get_yahoo_finance(ticker: str, indicator: str) -> Dict[str, Any]:
    try:
        if indicator == "price":
            data = yf.Ticker(ticker).history(period="1d")
            return {"ticker": ticker, "price": float(data['Close'].iloc[-1])}
        elif indicator == "per":
            info = yf.Ticker(ticker).info
            return {"ticker": ticker, "per": info.get('trailingPE', None)}
        elif indicator == "pbr":
            info = yf.Ticker(ticker).info
            return {"ticker": ticker, "pbr": info.get('priceToBook', None)}
        elif indicator == "dividend":
            info = yf.Ticker(ticker).info
            dividend = info.get('dividendYield', None)
            if dividend is not None:
                dividend = round(dividend * 100, 2)
            return {"ticker": ticker, "dividend": dividend}
        else:
            return {"error": "지원하지 않는 지표입니다"}
    except Exception as e:
        return {"error": f"야후 파이낸스 데이터 가져오기 실패: {str(e)}"}

def get_financial_data(ticker: str) -> Dict[str, Any]:
    # KRX 우선, 실패시 Yahoo로 fallback
    result = {"ticker": ticker}
    for indicator in ["price", "per", "pbr"]:
        value = get_pykrx_data(ticker, indicator)
        if "error" in value or value.get(indicator) is None:
            value = get_yahoo_finance(ticker, indicator)
        if indicator in value:
            result[indicator] = value[indicator]
        else:
            result[indicator] = None
    # 배당수익률은 Yahoo만
    dividend = get_yahoo_finance(ticker, "dividend")
    result["dividend"] = dividend.get("dividend") if "dividend" in dividend else None
    return FinancialData(**result).dict()
def get_financial_TimeSeries_data(ticker: str, nY: int=3) -> Dict[str, Any]:
    data = get_pykrx_data_nY(ticker, nY)
    # 배당수익률은 Yahoo만
    return Financial_TimeSeries_Data(**data).dict()

def get_all_tickers() -> pd.DataFrame:
    """
    KRX 상장 종목 코드와 이름을 반환합니다.
    """
    tickers = pykrx_stock.get_market_ticker_list(market="ALL")
    return {pykrx_stock.get_market_ticker_name(ticker):ticker for ticker in tickers}