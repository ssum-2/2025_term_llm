import os, base64
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.prompts import ChatPromptTemplate


def parse_pdf(pdf_path):
    elements = []
    print(f"[parse_pdf] 호출됨, pdf_path: {pdf_path}")
    doc = fitz.open(pdf_path)
    all_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            all_text += text + "\n\n"
        try:
            tables = page.find_tables()
            if tables and tables.tables:
                for t in tables:
                    md = t.to_markdown()
                    elements.append({'type': 'table', 'content': md, 'page': page_num + 1})
        except Exception:
            pass

        # 이미지 디렉토리 생성
        os.makedirs('./tmp/images', exist_ok=True)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0] if isinstance(img, tuple) else img
            base_image = doc.extract_image(xref)
            if base_image:
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = f"./tmp/images/tmp_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                elements.append({'type': 'image', 'content': image_path, 'page': page_num + 1})

    if all_text.strip():
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200, separators=["\n\n\n", "\n\n", "\n", ". ", " "]
        )
        split_texts = text_splitter.split_text(all_text)
        for idx, chunk in enumerate(split_texts):
            elements.append({'type': 'text', 'content': chunk, 'page': f"chunk_{idx + 1}"})

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
                for t in tables:
                    if t:
                        df = pd.DataFrame(t[1:], columns=t[0])
                        md = df.to_markdown(index=False)
                        elements.append({'type': 'table', 'content': md, 'page': page_num + 1})
            except Exception:
                pass

    print(f"[parse_pdf] 종료됨, elements 개수: {len(elements)}")
    return elements


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
def summarize_single_element(element: dict) -> Document | None:
    """인자로 받은 모델을 사용하여 요소를 요약합니다."""
    elem_type = element.get("type")
    content = element.get("content")
    page = element.get("page")
    mode = element.get("mode")
    pdf_path = element.get("pdf_path")
    # 인자에서 모델 가져오기
    llm: BaseChatModel = element.get("llm")
    openai_client: OpenAI | None = element.get("openai_client")

    if not llm:
        print(f"[ERROR] LLM instance not provided for summarization (Page: {page}).")
        return None

    source_file = os.path.basename(pdf_path)
    print(f"[{os.getpid()}] summarize_single_element: page {page} ({elem_type}) 요약 시작")
    summary_text = ""
    metadata = {"source_file": source_file, "page": page, "element_type": elem_type}

    try:
        if elem_type == "text":
            prompt = ChatPromptTemplate.from_template("다음은 금융 리포트의 일부입니다. 핵심 내용을 한두 문장으로 간결하게 요약하세요:\n\n{content}")
            chain = prompt | llm
            summary_text = chain.invoke({"content": content}).content

        elif elem_type == "table":
            prompt = ChatPromptTemplate.from_template("다음은 금융 리포트의 표입니다. 이 표가 의미하는 핵심 정보와 수치를 요약하세요:\n\n{content}")
            chain = prompt | llm
            summary_text = chain.invoke({"content": content}).content

        elif elem_type == "image":
            if mode != "openai" or not openai_client:
                summary_text = f"[INFO] Image captioning skipped (Mode: {mode}, Client: {'OK' if openai_client else 'Not OK'})."
            else:
                summary_text = caption_images(content, openai_client)

        else:
            print(f"알 수 없는 요소 타입 발견: {elem_type}")
            return None

    except Exception as e:
        print(f"[ERROR] LLM 요약 실패 (Page: {page}): {str(e)}")
        summary_text = f"[ERROR] LLM 요약 실패 (Page: {page}): {str(e)}"

    return Document(page_content=summary_text, metadata=metadata)


def caption_images(image_path: str, client: OpenAI) -> str:
    """인자로 받은 OpenAI 클라이언트를 사용하여 이미지 캡션을 생성합니다."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    prompt = "이 이미지를 설명하는 한글 캡션을 생성하세요. 금융/증권 리포트의 차트 또는 그래프일 가능성이 높습니다."
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
            ]}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content