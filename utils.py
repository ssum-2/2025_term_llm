import os, base64, requests, re, shutil
import fitz  # PyMuPDF
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional
from collections import defaultdict


# --- PDF 분할 클래스 ---
class PDFSplitter:
    @staticmethod
    def split(filepath: str, batch_size: int = 10, output_dir: str = "./tmp/split_pdfs") -> List[str]:
        # if os.path.exists(output_dir):
        #     shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        input_pdf = fitz.open(filepath)
        num_pages = len(input_pdf)
        print(f"📄 총 페이지 수: {num_pages}")

        split_files = []
        for start_page in range(0, num_pages, batch_size):
            end_page = min(start_page + batch_size, num_pages)
            input_file_basename = os.path.splitext(os.path.basename(filepath))[0]
            output_file = os.path.join(output_dir, f"{input_file_basename}_{start_page:04d}_{end_page - 1:04d}.pdf")

            with fitz.open() as output_pdf:
                output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page - 1)
                output_pdf.save(output_file)
                split_files.append(output_file)

        input_pdf.close()
        print(f"✅ PDF 분할 완료: {len(split_files)}개 파일 생성")
        return split_files


# --- Layout 분석 클래스 ---
class LayoutAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.upstage.ai/v1/document-ai/layout-analysis"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, pdf_path: str) -> Optional[Dict[str, Any]]:
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


# --- 요소 추출 및 처리 클래스 ---
class ElementProcessor:
    @staticmethod
    def extract_and_structure_elements(analysis_results: Dict[str, Dict]) -> Dict[str, Any]:
        page_elements = defaultdict(list)
        for split_file_path, result_json in analysis_results.items():
            match = re.search(r'_(\d{4})_(\d{4})\.json$', split_file_path)
            start_page_offset = int(match.group(1)) if match else 0

            for element in result_json.get("elements", []):
                original_page = element.get("page", 1)
                absolute_page = start_page_offset + original_page
                element['absolute_page'] = absolute_page
                page_elements[absolute_page].append(element)
        return dict(sorted(page_elements.items()))

    @staticmethod
    def crop_and_save_elements(pdf_path: str, page_elements: Dict[int, List[Dict]]) -> None:
        output_folder = os.path.join("./tmp/elements", os.path.splitext(os.path.basename(pdf_path))[0])
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        doc = fitz.open(pdf_path)
        for page_num, elements in page_elements.items():
            page = doc.load_page(page_num - 1)
            for elem in elements:
                if elem.get('category') in ['figure', 'table']:
                    bbox = elem['bounding_box']
                    x0, y0 = bbox[0]['x'], bbox[0]['y']
                    x1, y1 = bbox[2]['x'], bbox[2]['y']
                    rect = fitz.Rect(x0, y0, x1, y1)
                    clip_rect = rect & page.rect

                    if not clip_rect.is_empty:
                        pix = page.get_pixmap(clip=clip_rect, dpi=200)
                        output_path = os.path.join(output_folder, f"element_{page_num}_{elem['id']}.png")
                        pix.save(output_path)
                        elem['cropped_image_path'] = output_path
        doc.close()

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
    def summarize_element(element: Dict, context_text: str, llm: BaseChatModel,
                          openai_client: Optional[OpenAI] = None) -> str:
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