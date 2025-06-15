import os, base64, requests, json, re
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from PIL import Image
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from typing import List, Dict, Any, Optional


class LayoutAnalyzer:
    """Upstage Layout Analysis API를 호출하여 PDF의 구조를 분석하는 클래스"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.upstage.ai/v1/document-ai/layout-analysis"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """API를 호출하여 레이아웃 분석을 수행하고 결과를 JSON으로 반환합니다."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with open(pdf_path, "rb") as f:
                files = {"document": f}
                response = requests.post(self.api_url, headers=headers, files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ [API Error] Layout Analysis API 호출 실패 ({pdf_path}): {e}")
            return None


def crop_element_from_pdf(pdf_path: str, page_num: int, bbox: List[Dict[str, float]], output_folder: str,
                          element_id: str) -> str:
    """PDF 페이지에서 특정 bounding box 영역을 이미지로 잘라내 저장합니다."""
    output_path = os.path.join(output_folder, f"element_{element_id}.png")
    os.makedirs(output_folder, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)

        x0, y0 = bbox[0]['x'], bbox[0]['y']
        x1, y1 = bbox[2]['x'], bbox[2]['y']

        rect = fitz.Rect(x0, y0, x1, y1)
        clip_rect = rect & page.rect
        if not clip_rect.is_empty:
            pix = page.get_pixmap(clip=clip_rect, dpi=200)
            pix.save(output_path)
        else:
            return ""
        doc.close()
        return output_path
    except Exception as e:
        print(f"❌ [Cropping Error] Element {element_id} 이미지 자르기 실패: {e}")
        return ""


def parse_pdf_advanced(pdf_path: str, api_key: str) -> Dict[str, Any]:
    """Layout Analysis API를 사용하여 PDF를 파싱하고 구조화된 요소를 반환합니다."""
    print(f"🚀 [Parser] '{os.path.basename(pdf_path)}'의 초강력 분석 시작...")

    analyzer = LayoutAnalyzer(api_key)
    analysis_result = analyzer.analyze(pdf_path)
    if not analysis_result or "elements" not in analysis_result:
        print(f"⚠️ [Parser] '{os.path.basename(pdf_path)}' 분석 실패 또는 결과 없음.")
        return {"elements_by_page": {}, "author": "작성자"}

    elements_by_page = defaultdict(list)
    for element in analysis_result["elements"]:
        elements_by_page[element.get("page", 1)].append(element)

    author = "작성자"
    if 1 in elements_by_page:
        first_page_text = " ".join([elem.get('text', '') for elem in elements_by_page[1]])
        match = re.search(r'([\w\s]+(?:증권|투자증권|자산운용|경제연구소|리서치))', first_page_text)
        if match:
            author = match.group(1).strip().replace("\n", " ")

    output_folder = os.path.join("./tmp/elements", os.path.splitext(os.path.basename(pdf_path))[0])
    for page_num, elements in elements_by_page.items():
        for elem in elements:
            if elem.get('category') in ['figure', 'table']:
                cropped_path = crop_element_from_pdf(
                    pdf_path=pdf_path, page_num=page_num, bbox=elem['bounding_box'],
                    output_folder=output_folder, element_id=elem['id']
                )
                elem['cropped_image_path'] = cropped_path

    return {"elements_by_page": dict(elements_by_page), "author": author}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
def summarize_element_advanced(element: Dict, context_text: str, llm: BaseChatModel,
                               openai_client: Optional[OpenAI] = None) -> str:
    """구조화된 요소를 바탕으로 컨텍스트를 고려하여 요약문을 생성합니다."""
    category = element.get("category")

    try:
        if category in ['title', 'header', 'paragraph', 'list']:
            prompt = ChatPromptTemplate.from_template("다음은 금융 리포트의 일부입니다. 핵심 내용을 1-2 문장으로 간결하게 요약하세요:\n\n{content}")
            chain = prompt | llm
            return chain.invoke({"content": element.get('text', '')}).content

        elif category in ['figure', 'table']:
            image_path = element.get('cropped_image_path')
            if not image_path or not os.path.exists(image_path) or not openai_client:
                return f"[{category.capitalize()} 요약 생략: 이미지 파일 없음]"

            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            prompt_text = (f"당신은 금융 데이터 분석 전문가입니다. 다음은 리포트의 일부 텍스트 컨텍스트와 이미지입니다."
                           f"이미지({category})의 핵심 내용과 컨텍스트를 종합하여 가장 중요한 정보를 요약해주세요.\n\n"
                           f"## 주변 텍스트 컨텍스트:\n{context_text}\n\n"
                           f"## 분석 대상 이미지({category}):")

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]}],
                max_tokens=500
            )
            return response.choices[0].message.content

        else:
            return ""
    except Exception as e:
        print(f"❌ [{category} 요약 오류] {e}")
        return f"[{category} 요약 중 오류 발생]"
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