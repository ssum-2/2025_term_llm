# README.md

# 1. 환경구성

1. uv 설치
2. python package 설치
    
    ```bash
    uv pip install langchain langchain-upstage faiss-cpu python-dotenv PyPDF2 flask
    uv pip install yfinance pykrx
    ```
    - 실행 시 오류가 발생한다면 requirements.txt 또는 pyproject.toml 참고
   
3. 필요 API 발급 및 등록
    - .env
        
        ```bash
        UPSTAGE_API_KEY=up_***
        OPENAI_API_KEY=sk-* # 혹시나 필요할까봐
        ```
        
    - upstage api는 콘솔 > 대시보드 - API keys 에서 확인
        - [Console - Upstage](https://console.upstage.ai/api-keys?api=chat)
        - 학생 크레딧 가입
            - https://velog.io/@link_dropper/upstage-free-soloar-pro
            - [Upstage-AWS AI Initiative](https://www.upstage.ai/events/ai-initiative-2025-ko)

## 2. 실행방법

1. 폴더 이동
    
    ```python
    git clone https://github.com/ssum-2/2025_term_llm.git
    
    # 또는 파일 다운로드 후
    
    cd 2025_term_llm
    ```
    
2. [app.py](http://app.py) 실행
    
    ```bash
    uv run app.py
    ```
    
3. 앱 실행 후 실제 구동 화면 (리포트 업로드 → 질의 → 응답 회신)
![img](https://i.imgur.com/IfSBGWR.png)