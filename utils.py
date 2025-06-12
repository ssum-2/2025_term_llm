import os, time
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import base64
from openai import OpenAI
from langchain_upstage import ChatUpstage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


def parse_pdf(pdf_path):
    # 텍스트, 표, 이미지 모두 추출
    # 표는 PyMuPDF 우선, 실패시 pdfplumber fallback
    # 이미지는 PyMuPDF로 추출
    elements = []
    print(f"[parse_pdf] 호출됨, pdf_path: {pdf_path}")
    # 텍스트/표(PyMuPDF)
    doc = fitz.open(pdf_path)
    all_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            all_text += text + "\n"
        # table
        try:
            tables = page.find_tables()
            if tables and tables.tables:
                for t in tables:
                    md = t.to_markdown()
                    elements.append({'type': 'table', 'content': md, 'page': page_num + 1})
        except Exception:
            pass

        # image (PyMuPDF)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0] if isinstance(img, tuple) else img
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = f"./tmp/tmp_page{page_num + 1}_img{img_index + 1}.{image_ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            elements.append({'type': 'image', 'content': image_path, 'page': page_num + 1})

    # 전체 텍스트를 chunk_size 단위로 분할; 페이지 단위로 처리시 너무 오래 걸려 청크 사이즈대로 뭉쳐서 보냄
    if all_text.strip():
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # 원하는 크기로 조절 (예: 1000~1500)
            chunk_overlap=160,
            separators=["\n\n## ", "\n\n", "\n", " "]  # 섹션 기반 분할
        )
        split_texts = text_splitter.split_text(all_text)
        for idx, chunk in enumerate(split_texts):
            elements.append({'type': 'text', 'content': chunk, 'page': f"chunk_{idx + 1}"})

    # table (pdfplumber fallback)
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
def summarize_single_element(element: dict) -> Document | None:
    elem_type = element.get("type")
    content = element.get("content")
    page = element.get("page")
    mode = element.get("mode", "solar")

    pdf_path = element.get("pdf_path", "unknown_file")
    source_file = os.path.basename(pdf_path)

    print(f"[{os.getpid()}] summarize_single_element: page {page} ({elem_type}) 요약 시작")

    summary_text = ""
    # 공통 메타데이터 미리 정의
    metadata = {"source_file": source_file, "page": page, "element_type": elem_type}

    try:
        if elem_type == "text":
            prompt = f"다음은 금융 리포트의 일부입니다. 핵심 내용을 간결하게 요약하세요:\n\n{content}"
            if mode == "solar":
                llm = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model_name="solar-1-mini-chat")
            else:
                llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
            summary_text = llm.invoke(prompt).content

        elif elem_type == "table":
            prompt = f"다음은 금융 리포트의 표입니다. 이 표가 의미하는 핵심 정보를 요약하세요:\n\n{content}"
            if mode == "solar":
                llm = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model_name="solar-1-mini-chat")
            else:
                llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
            summary_text = llm.invoke(prompt).content

        elif elem_type == "image":
            if mode != "openai":
                summary_text = f"[INFO] Image captioning skipped for non-OpenAI model. (Page: {page})"
            else:
                summary_text = caption_images(content)

        else:
            print(f"알 수 없는 요소 타입 발견: {elem_type}")
            return None

    except Exception as e:
        print(f"[ERROR] LLM 요약 실패 (Page: {page}): {str(e)}")
        summary_text = f"[ERROR] LLM 요약 실패 (Page: {page}): {str(e)}"

    return Document(page_content=summary_text, metadata=metadata)
def caption_images(image_path: str) -> str:
    # 이미지 캡셔닝을 OpenAI Vision API만 사용하도록 구현
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = "이 이미지를 설명하는 한글 캡션을 생성하세요."
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