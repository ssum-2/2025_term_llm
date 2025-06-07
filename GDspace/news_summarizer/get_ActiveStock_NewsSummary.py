import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
import warnings
import re
import time
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from utils.Gemma_tool import get_Gemma_response,clean_gemma_output

warnings.filterwarnings('ignore')
def clean_text(article_text: str) -> str:
    # (A) HTML 태그 제거
    article_text = re.sub(r"<[^>]+>", "", article_text)

    # (B) URL 제거
    article_text = re.sub(r"http\S+|www\S+|https?:\S+", "", article_text)

    # (C) 기자명, (광고), (Copyright) 등 자주 등장하는 불필요문구 제거 예시
    # 패턴은 실제 기사들을 보고 맞춰서 수정 가능합니다.
    article_text = re.sub(r"기사 원문.*", "", article_text)
    article_text = re.sub(r"Copyright.*", "", article_text)

    # (D) 이모지 혹은 유니코드 특수문자 제거
    article_text = re.sub(r"[\U00010000-\U0010ffff]", "", article_text)

    # (E) 불필요한 중복 공백, 줄바꿈 등 정리
    article_text = re.sub(r"\s+", " ", article_text).strip()

    article_text = re.sub(r"\[.*?\]", "", article_text)

    return article_text
def get_ActiveStocks(stock_urls=[
                                 ['KR-invcom', "https://kr.investing.com/equities/most-active-stocks"],
                                 ['US-yahoo', "https://finance.yahoo.com/markets/stocks/most-active/?start=0&count=50"],
                                 ],
                     stock_watchlist={'US-yahoo': {'SATL': 'https://finance.yahoo.com/quote/SATL/',
                                                   'PL': 'https://finance.yahoo.com/quote/PL/',
                                                   'BKSY': 'https://finance.yahoo.com/quote/BKSY/'},
                                      'US-invcom': {'SATL': 'https://investing.com/equities/cf-acquisition-v',
                                                    'PL': 'https://investing.com/equities/planet-labs-pbc',
                                                    'BKSY': 'https://investing.com/equities/osprey-technology-acquisition'},
                                      'KR-invcom': {'SATL': 'https://kr.investing.com/equities/cf-acquisition-v',
                                                    'PL': 'https://kr.investing.com/equities/planet-labs-pbc',
                                                    'BKSY': 'https://kr.investing.com/equities/osprey-technology-acquisition'},},
                     wait_time=5
                     ):
    TickerLink_df = pd.DataFrame(columns=['Ticker', 'Link', 'Country', 'Source'])
    service = Service()

    for country_source, Stock_url in stock_urls:
        country, source = country_source.split('-')

        # Selenium 드라이버 열기
        driver = webdriver.Chrome(service=service)
        driver.get(Stock_url)
        time.sleep(wait_time)  # 페이지 로딩, Cloudflare 검사 대기

        # 페이지 소스 가져오기
        html = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html, "html.parser")

        # 사이트별 테이블 찾기
        if source == 'yahoo':
            table = soup.find("table", {"class": re.compile(r"^yf-[0-9a-z]+ bd$")})
        elif source == 'invcom':
            table = soup.find("table", {"class": "datatable-v2_table__93S4Y"})
        else:
            raise ValueError(f"[경고] 지원되지 않는 소스: {source}")


        if not table:
            raise ValueError(f"[경고] 테이블을 찾지 못했습니다. URL={Stock_url}")

        body = table.find("tbody")
        if not body:
            raise ValueError(f"[경고] 테이블의 tbody를 찾지 못했습니다. URL={Stock_url}")
        rows = body.find_all("tr")

        # 1) 실제 테이블에서 크롤링해온 데이터
        table_data = []
        for row in tqdm(rows, desc=f'Getting {country} stock news from {source}...', leave=False):
            a_tag = row.find("a")
            ticker = a_tag.text.strip()
            link = a_tag.get("href")
            if source == 'yahoo' and link.startswith("/quote/"):
                link = f"https://finance.yahoo.com{link}"
            table_data.append({"Ticker": ticker, "Link": link, "Country": country, "Source": source})
        table_df = pd.DataFrame(table_data)

        # 2) 워치리스트 (country_source) 추가
        watchlist_data = []
        if country_source in stock_watchlist:
            for t, l in stock_watchlist[country_source].items():
                watchlist_data.append({"Ticker": t, "Link": l, "Country": country, "Source": source})
        watchlist_df = pd.DataFrame(watchlist_data)

        # 3) 하나로 합치기
        tmp_df = pd.concat([watchlist_df, table_df], ignore_index=True)

        # 필요시 중복 제거
        tmp_df.drop_duplicates(subset=["Ticker", "Link"], inplace=True)

        # 결과에 누적
        TickerLink_df = pd.concat([TickerLink_df, tmp_df], ignore_index=True)

    # 최종 중복 제거 (여러 URL 간에도 중복되는 경우가 있을 수 있음)
    TickerLink_df.drop_duplicates(subset=["Ticker", "Link"], inplace=True)
    return TickerLink_df


if __name__ == "__main__":
    today_date = pd.Timestamp.now().normalize()
    today_date_str = today_date.strftime("%Y%m%d")
    # dayofweek: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6

    if today_date.dayofweek == 0:
        # 오늘이 월요일이면, 어제가 아니라 그제(2일 전)로 설정
        kr_yesterday_date = today_date - pd.Timedelta(days=2)
        us_yesterday_date = today_date - pd.Timedelta(days=2)
    else:
        # 그 외 요일이면 어제(1일 전)로 설정
        kr_yesterday_date = today_date - pd.Timedelta(days=1)
        us_yesterday_date = today_date

    # yesterday_date = pd.Timestamp("2025-03-10")
    headers = {
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.9"
        ),
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "DNT": "1",
        "Referer": "https://www.investing.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/112.0.5615.49 Safari/537.36"
        )
    }
    scraper = cloudscraper.create_scraper(browser='chrome')

    max_try = 5
    max_pages = 1
    max_articles=3
    news_list = []
    TickerLink_df=get_ActiveStocks()

    for ticker, stock_news_link, country, source in tqdm(TickerLink_df.values[::-1]):
        if source == 'invcom': max_pages = 2
        # if ticker=='NVDA':
        #     ㅇㅁㄴㅁ
        print(f'TICKER: {ticker}')
        ticker_article = 0
        for page_num in range(1, max_pages + 1):
            if source == 'yahoo':
                page_url = f"{stock_news_link}news"
            elif source == 'invcom':
                page_url = f"{stock_news_link}-news/{page_num}"
            else:
                raise ValueError("Invalid source")

            try:
                response = scraper.get(page_url, headers=headers)
            except:
                print(f"RETRY RESPONSE - {page_url}")
                time.sleep(10)
                response = scraper.get(page_url, headers=headers)

            try_cnt = 0
            Failed = 0
            while response.status_code != 200 and (try_cnt <= max_try):
                try_cnt += 1
                time.sleep(0 + (try_cnt * 2))
                print(f"retry {try_cnt}: {ticker} | page num: {page_num}")
                response = scraper.get(page_url, headers=headers)
                if try_cnt == max_try:
                    print(f"Failed to get {ticker} | page num: {page_num}")
                    Failed = 1
                    break
                    # raise ValueError("Failed to get the page")
            if Failed == 1:
                continue
            soup = BeautifulSoup(response.content, "html.parser")

            if source == 'yahoo':
                articles = soup.select('li[class="stream-item story-item yf-1drgw5l"]') # li[class="stream-item story-item yf-1usaaz9"]
            elif source == 'invcom':
                articles = soup.select('article[data-test="article-item"]')
            else:
                raise ValueError("Invalid source")
            MAX_ARTICLES_REACHED=False
            for article in articles:
                if MAX_ARTICLES_REACHED:
                    print(f"MAX ARTICLES REACHED FOR {ticker}")
                    continue
                # 제목/링크
                if source == 'yahoo':
                    title_tag = article.select_one('a.subtle-link.fin-size-small.titles.noUnderline')
                elif source == 'invcom':
                    title_tag = article.find('a', attrs={'data-test': 'article-title-link'})
                else:
                    raise ValueError("Invalid source")
                news_link = title_tag.get('href')
                if pd.isnull(news_link):
                    continue
                if not news_link.startswith('http'):
                    continue

                if source == 'yahoo':
                    title = title_tag.find('h3').get_text(strip=True)
                elif source == 'invcom':
                    title = title_tag.get_text(strip=True)
                else:
                    raise ValueError("Invalid source")

                if source == 'yahoo':
                    provider = article.select_one('div.publishing').get_text(strip=True).split('•')[0].strip()
                elif source == 'invcom':
                    provider = article.select_one('span[data-test="news-provider-name"]').get_text(strip=True)
                else:
                    raise ValueError("Invalid source")

                # if provider in 'Investing.com':
                # print('PASS: Investing.com news')
                # continue
                if source == 'yahoo':
                    publish_time = article.select_one('div.publishing').get_text(strip=True).split('•')[-1].strip()
                elif source == 'invcom':
                    publish_time = article.find('time', attrs={'data-test': 'article-publish-date'}).get_text(
                        strip=True)
                else:
                    raise ValueError("Invalid source")

                if ('년' in publish_time) & ('월' in publish_time) & ('일' in publish_time):
                    publish_time = pd.to_datetime(publish_time, format='%Y년 %m월 %d일').strftime("%Y-%m-%d")
                else:
                    publish_time = publish_time.replace('시간', ' hours').replace('분', ' minutes').replace('초', ' second').replace('일', ' days').replace('월', ' months')
                publish_time = publish_time.replace('in ', '')

                published_date = None  # pd.Timestamp를 담을 변수
                if ("minute" in publish_time or "minutes" in publish_time):
                    minutes_ago = int(publish_time.split()[0])
                    published_dt = pd.Timestamp.now() - pd.Timedelta(minutes=minutes_ago)
                    published_date = published_dt.normalize()  # 시분초 제거
                elif "hour" in publish_time or "hours" in publish_time:
                    hours_ago = int(publish_time.split()[0])
                    published_dt = pd.Timestamp.now() - pd.Timedelta(hours=hours_ago)
                    published_date = published_dt.normalize()
                elif "day" in publish_time or "days" in publish_time:
                    if publish_time.split()[0] == 'yesterday':
                        days_ago = 1
                    else:
                        days_ago = int(publish_time.split()[0])
                    published_dt = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
                    published_date = published_dt.normalize()
                elif "month" in publish_time or "months" in publish_time:
                    if publish_time.split()[0] == 'last':
                        months_ago = 1
                    else:
                        months_ago = int(publish_time.split()[0])

                    published_dt = pd.Timestamp.now() - pd.Timedelta(days=months_ago * 30)
                    published_date = published_dt.normalize()
                else:
                    # time_tag 에 datetime 속성이 있으면 우선 사용
                    datetime_str = article.find('time', attrs={'data-test': 'article-publish-date'}).get('datetime')
                    if datetime_str:
                        published_dt = pd.Timestamp(datetime_str)
                        published_date = published_dt.normalize()
                    else:
                        # 화면에 보이는 문자열(예: "Jan 30, 2025 21:23")을 직접 파싱할 수도 있음
                        # 그러나 investing.com 구조상 time_tag의 datetime="..." 가 대부분 존재하므로
                        # 여기서는 간단히 동일하게 pd.Timestamp()에 던져 보도록 함
                        published_dt = pd.Timestamp(publish_time)
                        published_date = published_dt.normalize()

                is_recent = False
                if published_date is not None:
                    if country == 'KR':
                        if published_date >= kr_yesterday_date:
                            is_recent = True
                    elif country == 'US':
                        if published_date >= us_yesterday_date:
                            is_recent = True
                if is_recent:
                    news_response = scraper.get(news_link, headers=headers)
                    news_soup = BeautifulSoup(news_response.content, "html.parser")
                    if source == 'yahoo':
                        # article_div = news_soup.select_one('div.body.yf-3qln1o')
                        # article_div = news_soup.select_one('p.yf-1090901')
                        if (article_tag := news_soup.find('article')):
                            container = article_tag

                        # 그다음 schema.org 마이크로데이터
                        elif (body_div := news_soup.find('div', itemprop='articleBody')):
                            container = body_div
                        else:
                            # 클래스명이 yf-xxxxx 형태인 <p> 모두 수집
                            ps = news_soup.find_all('p', class_=re.compile(r"^yf-[0-9a-z]+$"))

                            # 그래도 없으면, 텍스트 길이 기준 필터
                            if not ps:
                                all_ps = news_soup.find_all('p')
                                ps = [p for p in all_ps if len(p.get_text(strip=True)) > 50]
                            # 임시 div에 붙여서 container로 사용
                            container = BeautifulSoup('<div></div>', 'html.parser').div
                            for p in ps:
                                container.append(p)
                    elif source == 'invcom':
                        # article_div = news_soup.select_one('#article.article_container .article_WYSIWYG__O0uhw.article_articlePage__UMz3q')
                        container = news_soup.select_one('#article.article_container .article_WYSIWYG__O0uhw.article_articlePage__UMz3q')
                    else:
                        raise ValueError('Invalid source')

                    article_text = ""
                    if container:
                        # 불필요 태그 제거
                        tags_to_remove = [
                            'script', 'style', 'img', 'iframe', 'figure',
                            'figcaption', 'button', 'nav', 'header', 'footer',
                            'svg', 'path', 'ul', 'li'
                        ]
                        for t in tags_to_remove:
                            for match in container.find_all(t):
                                match.decompose()

                        # 텍스트만 추출 및 클린징
                        raw = container.get_text(separator="\n", strip=True)
                        article_text = clean_text(raw)
                    summary_text = ""
                    stt_time = time.time()
                    if (len(article_text)>10_000)|(len(article_text)<200):
                        # Too long article pass
                        print(f"TOO LONG or TOO SHORT ARTICLE: {ticker} | {title} | {len(article_text)}")
                        continue

                    if country == 'KR':
                        ko_prmt = (
                            "<start_of_turn>user\n"
                            "당신은 **전문 금융·투자 요약가**입니다.\n"
                            "[규칙]\n"
                            "1. 시스템·사용자 지침·원문을 **절대 복사하지 마십시오**.\n"
                            "2. **추론·가정 금지**: 기사에 명시된 정보만 사용하십시오.\n"
                            f"3. **{len(article_text) // 2}자 이내, 다음 우선순위·조건을 준수합니다.\n"
                            "   ■ 우선순위\n"
                            "     • **Primary**: 기사 핵심 종목·섹터의 지표\n"
                            "     • **Secondary**: 시장 지수 — 길이 여유가 있을 때만 포함\n"
                            "   ■ 조건부 형식\n"
                            "     (A) 지표와 원인 모두 **명시** → ‘지표 + 원인’ 한 문장, 다중 지표는 콤마로 구분\n"
                            "     (B) 지표만 있고 원인 **없음** → 지표 뒤에 **“원인 미공개”** 표기\n"
                            "     (C) 지표 **없음** → ‘주요 사건/결과(+원인)’만 기술\n"
                            "4. Secondary 지표도 (A)~(C) 규칙을 동일하게 적용하며, **숫자 생략 금지**.\n"
                            "5. 오직 요약 문장만 출력하고 `<think>` 등 내부 사고 과정을 노출하지 마십시오.\n\n"
                            "### 원문\n"
                            f"{article_text}\n"
                            "<end_of_turn>\n"
                            "<start_of_turn>model\n"
                        )
                        stt_time = time.time()
                        LLM_summary = get_Gemma_response(prompt=ko_prmt, stream=False)
                        summary_text = clean_gemma_output(LLM_summary)
                    elif country == 'US':
                        en_prmt = (
                            "<start_of_turn>user\n"
                            "You are a **professional finance-and-investment summarizer**.\n"
                            "[Rules]\n"
                            "1. **Never reproduce** instructions or original text verbatim.\n"
                            "2. **No invention or inference**—use only facts explicitly stated in the article.\n"
                           f"3. Summarize in **~{len(article_text) // 2} characters, following the priority & conditional logic below.\n"
                            "   ■ Priority\n"
                            "     • **Primary**: metrics for the focal company/sector\n"
                            "     • **Secondary**: broad market indexes — include only if length permits\n"
                            "   ■ Conditional format\n"
                            "     (A) Metric **and** cause present → pair **metric + cause** in one sentence; separate multiple metrics with commas\n"
                            "     (B) Metric present, cause **absent** → list metric(s) and append **“cause not stated.”**\n"
                            "     (C) No metrics → state the **key event/outcome**; add the cause if provided\n"
                            "4. Apply rules (A)–(C) to Secondary metrics as well and **never omit numeric values**.\n"
                            "5. Output **only the summary sentence(s)**; never reveal `<think>` or chain-of-thought.\n\n"
                            "### Original\n"
                           f"{article_text}\n"
                            "<end_of_turn>\n"
                            "<start_of_turn>model\n"
                        )
                        stt_time = time.time()
                        LLM_summary = get_Gemma_response(prompt=en_prmt, stream=False)
                        summary_text = clean_gemma_output(LLM_summary)

                    print(
                        f'article_text:\n{article_text}\n\n'
                        f'summary_text:\n{summary_text}\n\n'
                        f"LLM_summary time: {time.time() - stt_time}"
                    )

                    if len(article_text) > 10:
                        ticker_article += 1
                        news_list.append({
                            "ticker": ticker,
                            "country": country,
                            "provider": provider,
                            "publish_time": publish_time,
                            "publish_date": published_date,
                            "title": title,
                            "link": news_link,
                            "raw_article": article_text,
                            "summary": summary_text
                        })
                if ticker_article>=max_articles:
                    MAX_ARTICLES_REACHED = True
    STOCK_news_df = pd.DataFrame(news_list)

    # save_loc = ''
    # os.makedirs(save_loc, exist_ok=True)
    # save_name = f'{save_loc}/ActiveStocksNews_{today_date_str}.hd5'
    # STOCK_news_df.to_parquet(save_name)