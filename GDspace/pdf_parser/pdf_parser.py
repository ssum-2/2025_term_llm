# pdf_parser.py
import io
import os
import time
import json
import requests
import PyPDF2
import hashlib
import base64
import re
from PIL import Image
from pathlib import Path
import markdown as md
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.Gemma_tool import get_Gemma_response, clean_gemma_output
from utils.NLP_tool import split_sentences
from typing import List, Dict, Any, Optional, Tuple, Union

# 문서 요소 클래스
class Element:
    def __init__(
        self,
        category: str,
        content: str,
        html: str,
        markdown: str,
        page: int,        id: int,
        base64_encoding: Optional[str] = None,
        image_filename: Optional[str] = None,
        coordinates: Optional[dict] = None
    ):
        self.category = category
        self.content = content
        self.html = html
        self.markdown = markdown
        self.page = page
        self.id = id
        self.base64_encoding = base64_encoding
        self.image_filename = image_filename
        self.coordinates = coordinates

        # 요약 결과 저장용 필드(추가)
        self.summary: Optional[str] = None     # 최종 요약(이미지+주변 텍스트 종합용)
        self.image_desc: Optional[str] = None  # 이미지 자체 설명(1회성)

    def __repr__(self):
        return f"Element(id={self.id}, category='{self.category}', page={self.page})"
class IncrementalPdfSummarizer:
    """
    - PDF 파싱 결과로부터 '점진적 요약'을 수행하는 클래스
    - 이미지(도표·차트 등) 요소는 Gemma_tool을 통해 한 번 설명받고,
      주변 텍스트와 합쳐 최종 요약으로 반영
    """

    def __init__(
            self,
            max_workers: int = 1,
            verbose: bool = True,
            gemma_model: str = None,  # Gemma_tool 모델명 (None 시 defaults)
    ):
        """
        Args:
            max_workers: 이미지 요약 등 병렬 처리 시 사용할 최대 스레드 수
            verbose: 로그 출력 여부
            text_model: NLP_tool.get_summary() 에서 사용할 언어 구분
            gemma_model: Gemma_tool의 model_id (None 시 Gemma_tool 내부 default 사용)
            use_gemma_for_text: True 시 텍스트 요약도 Gemma_tool.get_Gemma_response() 사용
                                False 시 NLP_tool.get_summary() 사용
        """
        self.max_workers = max_workers
        self.verbose = verbose
        self.gemma_model = gemma_model

    def log(self, msg: str):
        if self.verbose:
            print(f"[IncrementalPdfSummarizer] {msg}")

    def _summarize_text(self, text: str) -> str:
        """
        순수 텍스트(문장 수천자) 요약 함수.
        - NLP_tool 또는 Gemma_tool 중 선택적으로 사용
        """
        if not text.strip():
            return ""

        # (수정된) Gemma_tool 프롬프트
        # - 원문 그대로 복붙X, 추측/생성X, 핵심 요약
        # - 필요하다면 문장 수 제한/형식 지시 추가
        prompt = (
                  "<system>\n"
                  "<bos><start_of_turn>system\n"
                  "당신은 전문 금융 분석가 입니다. "
                  "원문을 그대로 복사하거나 새로운 정보를 추가하지 말고, "
                  "투자 의사결정에 필요한 핵심만 간결하게 정리하십시오.\n"
                  "<end_of_turn>\n"
                  "<start_of_turn>user\n"
                  "다음 원문을 한국어로 요약하세요:\n\n"
                  f"{text}\n"
                  "<end_of_turn>\n"
                  "<start_of_turn>model\n"
                  )
        stt_time = time.time()
        ans = get_Gemma_response(prompt = prompt, model_id = self.gemma_model, stream = False)
        ans = clean_gemma_output(ans)
        print(f'ans response: {ans}')
        print(f'ans response consume {time.time() - stt_time:.2f}s')
        return ans.strip()


    def _describe_image(self, image_path: str) -> str:
        """
        Gemma_tool을 이용해 이미지에 대한 1회성 설명(팩트 위주) 생성
        """
        if not image_path or not os.path.isfile(image_path):
            return ""

        # 아래 Prompt는 예시이므로, 주어진 가이드에 맞춰 수정 가능:
        # "절대로 추측하거나 추론하지 말 것. 주어진 정보만 설명하라" 등등
        prompt = (
                  "<bos><start_of_turn>system\n"
                  "당신은 이미지에서 보이는 사실만 기술하는 분석가입니다. "
                  "추측·배경지식·의견을 포함하지 마십시오.\n"
                  "<end_of_turn>\n"
                  "<start_of_turn>user\n"
                  "이 이미지에 대해 자세히 설명해 주세요\n"
                  "<end_of_turn>\n"
                  "<start_of_turn>model\n"
                  )
        stt_time = time.time()
        desc = get_Gemma_response(prompt=prompt, image_path=image_path, model_id=self.gemma_model)
        desc = clean_gemma_output(desc)
        print(f'desc response: {desc}')
        print(f'desc response consume {time.time() - stt_time:.2f}s')
        return desc.strip()

    def _integrate_image_with_context(self, image_desc: str, context_text: str) -> str:
        """
        이미지 설명 + 주변 텍스트를 종합하여 최종 이미지 summary를 생성
        """
        if (not image_desc.strip()) and (not context_text.strip()):
            return ""

        prompt = (
                  "<bos><start_of_turn>system\n"
                  "이미지 설명과 주변 텍스트를 결합해 핵심 의미만 간결히 요약하세요. "
                  "새로운 추론이나 상상은 금지합니다.\n"
                  "<end_of_turn>\n"
                  "<start_of_turn>user\n"
                  "이미지 설명:\n"
                  f"{image_desc}\n\n"
                  "주변 텍스트:\n"
                  f"{context_text}\n\n"
                  "두 정보를 통합해 3문장 이하로 요약해 주세요.\n"
                  "<end_of_turn>\n"
                  "<start_of_turn>model\n"
                  )
        stt_time = time.time()
        combined = get_Gemma_response(prompt = prompt, model_id = self.gemma_model, stream = False)
        combined = clean_gemma_output(combined)
        print(f'combined response: {combined}')
        print(f'combined response consume {time.time() - stt_time:.2f}s')
        return combined.strip()

    def _extract_text_by_page(self, elements: List[Element]) -> Dict[int, str]:
        """
        페이지별 텍스트 요소를 한데 모아 딕셔너리로 반환
        """
        page_text_map = {}
        for elem in elements:
            if elem.category not in ["paragraph", "caption", "list", "heading1", "equation", "index"]:
                continue
            p = elem.page
            if p not in page_text_map:
                page_text_map[p] = []
            page_text_map[p].append(elem.content.strip())

        # page_text_map[page] = "모든 텍스트"
        for p in page_text_map:
            page_text_map[p] = "\n".join(page_text_map[p]).strip()
        return page_text_map

    def _split_text_half(self, text: str) -> (str, str):
        """
        텍스트를 대략 반으로 나누어 (앞부분, 뒷부분) 반환
        - 개략적 구현
        """
        lines = split_sentences(text)
        if not lines:
            return (text, "")
        half = len(lines) // 2
        front = "\n".join(lines[:half])
        back = "\n".join(lines[half:])
        return (front, back)

    def _incremental_page_summarization(self, page_text_map: Dict[int, str]) -> Dict[int, str]:
        """
        1) 각 페이지 전체 요약
        2) (p후반 + p+1전반) 교차 요약
        3) 페이지별 최종 요약 = (전체 요약 + 해당 교차 요약) 재요약
        """
        sorted_pages = sorted(page_text_map.keys())

        # (A) 각 페이지 '전체 텍스트' 요약을 먼저 구해둠
        entire_summaries = {}
        for p in sorted_pages:
            raw_text = page_text_map[p]
            if raw_text.strip():
                entire_summaries[p] = self._summarize_text(raw_text)
            else:
                entire_summaries[p] = ""

        # (B) 교차 요약: (page i 후반 + page i+1 전반)
        cross_summaries = {}
        for i in range(len(sorted_pages) - 1):
            p = sorted_pages[i]
            p_next = sorted_pages[i + 1]

            # 텍스트 절반 분할
            p_front, p_back = self._split_text_half(page_text_map[p])
            n_front, n_back = self._split_text_half(page_text_map[p_next])

            cross_chunk = f"{p_back}\n\n{n_front}".strip()
            if cross_chunk:
                # 교차 구간 요약
                cross_summaries[p] = self._summarize_text(
                    f"아래 내용은 페이지 {p} 후반부와 페이지 {p_next} 초반부입니다.\n"
                    f"연결되는 중요한 내용을 간략히 요약해 주세요:\n\n{cross_chunk}"
                )
            else:
                cross_summaries[p] = ""

        # (C) 페이지별 '최종 요약' 결정
        final_page_summaries = {}
        for idx, p in enumerate(sorted_pages):
            # 기본은 '전체 요약'
            combined = entire_summaries[p]

            # 만약 다음 페이지와의 교차 요약이 있다면 → 같이 합쳐 재요약
            cross_part = cross_summaries.get(p, "")
            if cross_part.strip():
                # 전체 + 교차 구문 붙여서 다시 요약
                merged_text = (
                    f"[페이지 {p} 전체 요약]\n{combined}\n\n"
                    f"[페이지 {p} 후반+{p + 1} 전반 교차 요약]\n{cross_part}"
                )
                combined = self._summarize_text(merged_text)

            final_page_summaries[p] = combined

        return final_page_summaries

    def _summarize_images(
            self,
            elements: List[Element],
            page_text_map: Dict[int, str],
    ) -> None:
        """
        이미지(figure, chart, table 등)에 대해
        1) 이미지 자체 설명 -> elem.image_desc
        2) 이미지 주변 텍스트와 합쳐 최종 summary -> elem.summary

        - 주변 텍스트: "같은 페이지 텍스트"에서 이미지 좌표 근접 부분을 발췌하거나,
          간단히 "전체 페이지 텍스트"를 사용해도 됨.
        - 여기서는 예시로 "전체 페이지 텍스트"를 사용. 실제 구현에선 coordinates로
          주변 문장만 발췌할 수도 있음.
        """
        # 1) 이미지들을 병렬로 "이미지 자체 설명" 얻기
        #    (Gemma_tool에 동일하게 요청)
        images_to_process = [e for e in elements if e.category in ("figure", "chart", "table") and e.image_filename]
        self.log(f"총 {len(images_to_process)}개 이미지 요약(기본 설명) 시작")

        def _desc_job(elem: Element) -> (int, str):
            img_desc = self._describe_image(elem.image_filename)
            return (elem.id, img_desc)

        results_map = {}
        future_to_elem = {}  # future와 Element를 매핑
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            # 1) 작업들을 futures 리스트에 담고
            futures_list = []
            for e in images_to_process:
                fut = exe.submit(_desc_job, e)
                futures_list.append(fut)
                future_to_elem[fut] = e

            # 2) as_completed()를 tqdm로 감싸 진행도 표시
            for fut in tqdm(as_completed(futures_list),
                            total=len(futures_list),
                            desc="Image captioning"):
                eid, img_desc = fut.result()
                results_map[eid] = img_desc

        # 이후 results_map을 통해 image_desc를 할당
        for elem in images_to_process:
            elem.image_desc = results_map.get(elem.id, "")
            page_txt = page_text_map.get(elem.page, "")
            final_summary = self._integrate_image_with_context(elem.image_desc, page_txt)
            elem.summary = final_summary  # 최종 요약
            self.log(f"[Image summary] Element({elem.id}, page={elem.page}) 요약 완료")

    def summarize_document(self, parsed_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        실제로 PDF 전체를 요약하는 핵심 메서드

        parsed_results 예시 구조:
        {
            "metadata": [...],
            "elements": [Element(...), ...],
            "total_cost": float,
            ...
        }

        Returns:
            {
                "page_summaries": {페이지번호: 요약문, ...},
                "elements": [Element(...)],  # image/table은 elem.summary 포함
                "final_summary":  "문서 전체 요약문"
            }
        """
        elements = parsed_results.get("elements", [])
        if not elements:
            return {
                "page_summaries": {},
                "elements": [],
                "final_summary": ""
            }

        # (1) 페이지별 텍스트 수집
        page_text_map = self._extract_text_by_page(elements)

        # (2) **이미지 요약을 먼저 수행** (이미지 캡셔닝 + 주변 텍스트 종합)
        self._summarize_images(elements, page_text_map)

        if not page_text_map:
            # 텍스트가 거의 없는 경우(이미지만 있는 문서 등)라면, 여기서 종료
            return {
                "page_summaries": {},
                "elements": elements,
                "final_summary": "텍스트가 없는 문서입니다."
            }

        # (3) **페이지 순서대로 점진 요약** (“1페이지 전체 → (1후반+2전반) → 2페이지 전체 → …”)
        page_summaries = self._incremental_page_summarization(page_text_map)

        # (4) 페이지 요약 전부 합쳐 최종 문서 요약
        all_pages_summary_text = "\n\n".join([
            f"[Page {p}] {page_summaries[p]}" for p in sorted(page_summaries.keys())
        ])
        final_summary = self._summarize_text(all_pages_summary_text)

        # (5) 결과 리턴
        return {
            "page_summaries": page_summaries,
            "elements": elements,  # 이미지 elem.summary 등 반영됨
            "final_summary": final_summary
        }

# 유틸리티 함수: base64 인코딩된 이미지를 파일로 저장
def save_image_from_base64(base64_data: str, output_dir: str, category: str, page: int, index: int) -> Optional[str]:
    """
    base64 인코딩된 이미지를 PNG 파일로 저장합니다.
    
    Args:
        base64_data: base64 인코딩된 이미지 데이터
        output_dir: 출력 디렉토리
        category: 이미지 카테고리 (figure, chart, table 등)
        page: 페이지 번호
        index: 인덱스 번호
        
    Returns:
        저장된 이미지 파일의 경로 또는 오류 시 None
    """
    try:
        # base64 디코딩
        image_data = base64.b64decode(base64_data)

        # 바이트 데이터를 이미지로 변환
        image = Image.open(io.BytesIO(image_data))

        # output_dir 내에 images 폴더와 하위 카테고리 폴더 생성
        image_dir = os.path.join(output_dir, "images", category)
        os.makedirs(image_dir, exist_ok=True)
        
        # 원본 폴더명 추출
        original_name = os.path.basename(output_dir)
        
        # 해시 기반의 짧은 파일명 생성
        hash_obj = hashlib.md5((f"{original_name}_{page}_{index}").encode())
        hash_str = hash_obj.hexdigest()[:8]  # 8자 해시값만 사용
        
        # 간단한 파일명 생성
        short_filename = f"{hash_str}_{category[0]}_{page+1}_{index}.png"
        
        # 저장 경로 생성
        image_path = os.path.join(image_dir, short_filename)
        
        # 이미지 저장
        image.save(image_path)
        return image_path
            
    except Exception as e:
        print(f"[Save Image] 이미지 저장 중 오류 발생: {str(e)}")
        return None
def _normalize(text: str) -> str:
    # 1) 단락(두 줄 이상 띄움)은 보존하고
    text = re.sub(r'\n{2,}', '\n\n', text)        # 두 줄 이상은 그대로
    # 2) 나머지 1줄 개행은 공백으로
    text = re.sub(r'\n(?!\n)', ' ', text)

    # 3) 공백이 있든 없든, ·ㆍ•∙ 뒤에 공백이 오면 => 새 줄 + '* '
    text = re.sub(r'\s*[·•∙]\s+', '\n* ', text)   # ✨ 바뀐 부분
    return text.strip()

class DocumentProcessor:
    """
    PDF 문서를 분석하여 구조화된 데이터(Element 목록)로 변환하고,
    Upstage Document Parse API를 통해 OCR/구조 인식을 수행.
    """

    def __init__(self, api_key: str, config: dict, batch_size: int = 100, use_ocr: bool = True, save_images: bool = True, verbose: bool = True):
        """
        Args:
            api_key: Upstage API 키
            config: API에 넘길 기본 파라미터 dict
            batch_size: PDF 분할 시 한 파일당 최대 페이지 수
            use_ocr: OCR 사용 여부
            save_images: 이미지 저장 여부
            verbose: 로깅 여부
        """
        self.api_key = api_key
        self.config = config.copy()
        self.config["ocr"] = use_ocr

        self.batch_size = batch_size
        self.save_images = save_images
        self.verbose = verbose
        self.accumulated_cost = 0  # 인식 비용(예시)

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[DocumentProcessor] {message}")

    def split_pdf(self, filepath: str) -> List[str]:
        """
        큰 PDF를 여러 개의 작은 PDF로 분할.
        """
        output_folder = os.path.splitext(os.path.basename(filepath))[0]
        output_dir = os.path.join(Path(filepath).parent, output_folder)
        os.makedirs(output_dir, exist_ok=True)
        self.log(f"출력 디렉토리 생성: {output_dir}")

        try:
            with open(filepath, 'rb') as file:
                input_pdf = PyPDF2.PdfReader(file)
                num_pages = len(input_pdf.pages)
                self.log(f"파일의 전체 페이지 수: {num_pages} pages.")

                split_filepaths = []
                for start_page in range(0, num_pages, self.batch_size):
                    end_page = min(start_page + self.batch_size, num_pages) - 1

                    output_filename = f"split_{start_page:04d}_{end_page:04d}.pdf"
                    output_file = os.path.join(output_dir, output_filename)

                    output_pdf = PyPDF2.PdfWriter()
                    for page_num in range(start_page, end_page + 1):
                        output_pdf.add_page(input_pdf.pages[page_num])

                    with open(output_file, 'wb') as ofs:
                        output_pdf.write(ofs)

                    split_filepaths.append(output_file)
                    self.log(
                        f"분할 PDF 생성 완료: {os.path.basename(output_file)} "
                        f"(페이지 {start_page}-{end_page})"
                    )

            return split_filepaths

        except Exception as e:
            self.log(f"PDF 분할 중 오류 발생: {str(e)}")
            return []

    def parse_start_end_page(self, filepath: str) -> Tuple[int, int]:
        """
        split_0000_0009.pdf 같은 파일명에서 (start_page, end_page) 추출
        """
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        if not name_without_ext.startswith("split_") or len(name_without_ext) < 15:
            return (-1, -1)

        page_numbers = name_without_ext[6:]  # 0000_0009
        if (
                len(page_numbers) == 9
                and page_numbers[4] == "_"
                and page_numbers[:4].isdigit()
                and page_numbers[5:].isdigit()
        ):
            start_page = int(page_numbers[:4])
            end_page = int(page_numbers[5:])
            if start_page <= end_page:
                return (start_page, end_page)
        return (-1, -1)

    def analyze_document(self, filepath: str) -> Dict[str, Any]:
        """
        Upstage 문서 분석 API 호출 후 결과(json) 반환
        """
        # [추가] 먼저 기존에 분석된 JSON 결과가 있으면 바로 로드
        json_path = os.path.splitext(filepath)[0] + ".json"
        if os.path.exists(json_path):
            self.log(f"이미 분석된 결과(JSON)를 발견: {json_path} → API 호출 생략")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with open(filepath, "rb") as f:
                files = {"document": f}
                response = requests.post(
                    "https://api.upstage.ai/v1/document-ai/document-parse",
                    headers=headers,
                    data=self.config,  # OCR/좌표/출력형식 등
                    files=files,
                )

            if response.status_code == 200:
                data = response.json()

                # .json 파일로도 저장
                output_file = os.path.splitext(filepath)[0] + ".json"
                with open(output_file, "w", encoding="utf-8") as wf:
                    json.dump(data, wf, ensure_ascii=False)

                return data
            else:
                raise ValueError(
                    f"API 요청 실패 [{response.status_code}]: {response.text}"
                )

        except Exception as e:
            raise ValueError(f"문서 분석 중 오류 발생: {str(e)}")

    def parse_documents(self, filepaths: List[str]) -> Dict[str, Any]:
        """
        여러 PDF split 파일들에 대해 API 분석 수행 → 전체 Element 통합
        """
        if not filepaths:
            self.log("분석할 파일이 없습니다.")
            return {"metadata": [], "elements": [], "total_cost": 0}

        self.log(f"총 {len(filepaths)}개 파일 분석 시작")
        all_metadata = []
        all_elements = []
        pages_count = 0

        original_dir = None
        if filepaths:
            original_dir = os.path.dirname(os.path.dirname(filepaths[0]))

        for i, filepath in enumerate(filepaths):
            try:
                t0 = time.time()
                self.log(f"파일 분석 중({i + 1}/{len(filepaths)}): {filepath}")

                data = self.analyze_document(filepath)
                start_page, _ = self.parse_start_end_page(filepath)
                page_offset = start_page if start_page != -1 else 0

                # page/ID 보정
                for e in data["elements"]:
                    e["page"] += page_offset

                usage_info = data.get("usage", {})
                if "pages" in usage_info:
                    pages_count += int(usage_info["pages"])

                all_metadata.append({
                    "filepath": filepath,
                    "api": data.get("api"),
                    "model": data.get("model"),
                    "usage": data.get("usage"),
                })
                all_elements.extend(data["elements"])

                dt = time.time() - t0
                self.log(f"→ 완료: {dt:.2f}초 소요")

            except Exception as e:
                self.log(f"[오류] {filepath}: {str(e)}")
                continue

        # 비용 계산 (예시)
        current_cost = pages_count * 0.01
        self.accumulated_cost += current_cost
        self.log(f"현재 분석 비용: ${current_cost:.2f}, 누적: ${self.accumulated_cost:.2f}")

        processed_elements = []
        id_counter = 0

        # Element 변환
        for raw in all_elements:
            cat = raw["category"]
            if cat in ["footnote", "header", "footer"]:
                continue

            image_filename = None
            if (
                    self.save_images
                    and cat in ["figure", "chart", "table"]
                    and "base64_encoding" in raw
            ):
                if original_dir:
                    image_filename = save_image_from_base64(
                        raw["base64_encoding"],
                        original_dir,
                        cat,
                        raw["page"],
                        id_counter
                    )

            elem = None
            if cat == "equation":
                elem = Element(
                    category=cat,
                    content=raw["content"]["markdown"] + "\n",
                    html=raw["content"]["html"],
                    markdown=raw["content"]["markdown"],
                    page=raw["page"],
                    id=id_counter,
                    coordinates=raw.get("coordinates"),
                )
            elif cat in ["table", "figure", "chart"]:
                elem = Element(
                    category=cat,
                    content=raw["content"]["markdown"] + "\n",
                    html=raw["content"]["html"],
                    markdown=raw["content"]["markdown"],
                    base64_encoding=raw.get("base64_encoding"),
                    image_filename=image_filename,
                    page=raw["page"],
                    id=id_counter,
                    coordinates=raw.get("coordinates"),
                )
            elif cat == "heading1":
                elem = Element(
                    category=cat,
                    content=f'# {raw["content"]["text"]}\n',
                    html=raw["content"]["html"],
                    markdown=raw["content"]["markdown"],
                    page=raw["page"],
                    id=id_counter,
                    coordinates=raw.get("coordinates"),
                )
            elif cat in ["caption", "paragraph", "list", "index"]:
                clean_text = _normalize(raw["content"]["text"])
                elem = Element(
                    category=cat,
                    content=clean_text + "\n",
                    html=raw["content"]["html"],
                    markdown=clean_text,
                    page=raw["page"],
                    id=id_counter,
                    coordinates=raw.get("coordinates"),
                )

            if elem is not None:
                processed_elements.append(elem)
                id_counter += 1

        self.log(f"→ 변환된 Element: {len(processed_elements)}개")

        return {
            "metadata": all_metadata,
            "elements": processed_elements,
            "total_cost": current_cost,
            "accumulated_cost": self.accumulated_cost
        }

    def process_pdf(self, filepath: str) -> Dict[str, Any]:
        """
        PDF → split → parse_documents → 통합 결과 반환
        """
        self.log(f"PDF 처리 시작: {filepath}")
        split_files = self.split_pdf(filepath)
        if not split_files:
            return {"elements": [], "total_cost": 0, "error": "PDF 분할 실패"}

        result = self.parse_documents(split_files)
        self.log(f"PDF 처리 완료: {filepath}")
        return result
    
def _element_sort_key(elem: Element):
    """
    같은 페이지 안에서 원본의 시각적 순서를 최대한 보존하기 위한 정렬 키
    1) page → 2) y1 → 3) x1 → 4) id
    좌표가 없을 때는 id만 사용합니다.
    """
    if elem.coordinates and isinstance(elem.coordinates, dict):
        return (
            elem.page,
            elem.coordinates.get("y1", 0),
            elem.coordinates.get("x1", 0),
            elem.id,
        )
    return (elem.page, elem.id)
def reconstruct_to_markdown(elements: List[Element], output_dir: Union[str, Path], filename: str = "reconstructed.md") -> Path:
    """
    Element 객체를 Markdown 으로 재구성하고 파일로 저장
    """
    # 페이지별 정렬
    elements_sorted = sorted(elements, key=_element_sort_key)

    # 페이지별로 구분선 추가
    md_lines = []
    current_page = None
    for elem in elements_sorted:
        if current_page is None or elem.page != current_page:
            # 새 페이지 시작
            if current_page is not None:
                md_lines.append("\n---\n")  # 페이지 구분선
            md_lines.append(f"\n<!-- Page {elem.page + 1} -->\n")
            current_page = elem.page

        # 이미지 → 이미지 파일 링크, 그 외 → markdown
        if elem.category in ("figure", "chart", "table") and elem.image_filename:
            rel = os.path.relpath(elem.image_filename, output_dir)
            md_lines.append(f"![{elem.category}_{elem.id}]({rel})\n")
            md_lines.append(elem.markdown + "\n")  # 표 캡션/설명 등
        else:
            md_lines.append(elem.markdown + "\n")

    # 저장
    output_path = Path(output_dir) / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)

    return output_path
def reconstruct_to_html(elements: List[Element], output_dir: Union[str, Path], filename: str = "reconstructed.html") -> Path:
    """
    Element 객체를 HTML 로 재구성하고 파일로 저장
    """
    elements_sorted = sorted(elements, key=_element_sort_key)

    # 간단한 스타일 – 페이지마다 구분선
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='ko'><head>",
        "<meta charset='utf-8'/>",
        "<title>Reconstructed Document</title>",
        "<style>",
        "body{font-family: Pretendard, Arial, sans-serif; line-height:1.6; margin:40px;}",
        ".page{margin-bottom:80px; border-bottom:1px dashed #ccc; padding-bottom:40px;}",
        ".page h1,.page h2,.page h3{margin-top:1.2em;}",
        ".img-wrapper{text-align:center;margin:1em 0;}",
        ".img-wrapper img{max-width:100%;height:auto;border:1px solid #eee;}",
        "</style></head><body>",
    ]

    current_page = None
    for elem in elements_sorted:
        if current_page is None or elem.page != current_page:
            # 새 페이지 div 시작
            if current_page is not None:
                html_lines.append("</div>")  # 이전 page div 닫기
            html_lines.append(f"<div class='page' id='page-{elem.page + 1}'>")
            html_lines.append(f"<h2 style='margin-top:0'>Page {elem.page + 1}</h2>")
            current_page = elem.page

        # 콘텐츠 삽입
        if elem.category in ("figure", "chart") and elem.image_filename:
            rel = os.path.relpath(elem.image_filename, output_dir)
            html_lines.append("<div class='img-wrapper'>")
            html_lines.append(f"<img src='{rel}' alt='{elem.category}_{elem.id}' />")
            # 설명 있는 경우
            if elem.markdown.strip():
                cap_html = md.markdown(elem.markdown)
                html_lines.append(cap_html)
            html_lines.append("</div>")
        elif elem.category == "table":
            # HTML 버전이 가장 레이아웃이 정확
            table_html = elem.html if elem.html else md.markdown(elem.markdown)
            html_lines.append(table_html)
        elif elem.category in ("caption", "paragraph", "list", "index"):  # 🔴 새 조건
            content_html = md.markdown(elem.markdown)  # 항상 markdown → HTML 변환
            html_lines.append(content_html)
        else:
            content_html = elem.html if elem.html else md.markdown(elem.markdown)
            html_lines.append(content_html)

    if current_page is not None:
        html_lines.append("</div>")  # 마지막 page div 닫기
    html_lines.append("</body></html>")

    output_path = Path(output_dir) / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(html_lines)

    return output_path

if __name__ == "__main__":
    pdf_path = "./20250228===모든 것은 계획대로===최민하,강영훈===삼성증권===기업분석.pdf"
    UPSTAGE_API_KEY = ''

    # 기본 설정 상수
    DEFAULT_CONFIG = {
        "ocr": True,  # OCR 사용 여부
        "coordinates": True,  # 좌표 정보 포함 여부
        "output_formats": "['html', 'text', 'markdown']",  # 출력 형식
        "model": "document-parse",  # 사용할 모델
        "base64_encoding": "['figure', 'chart', 'table']",  # base64로 인코딩할 요소
    }

    # 문서 처리 객체 생성
    processor = DocumentProcessor(
        api_key=UPSTAGE_API_KEY,  # API 키
        config = DEFAULT_CONFIG,
        batch_size=100,  # 한 번에 처리할 페이지 수
        use_ocr=True,  # OCR 사용 여부
        save_images=True,  # 이미지 저장 여부
        verbose=True  # 상세 로그 출력 여부
    )

    # PDF 파일을 작은 단위로 분할
    split_files = processor.split_pdf(pdf_path)

    # 분할된 PDF 파일 분석
    parsed_results = processor.parse_documents(split_files)

    # html_path = reconstruct_to_html(parsed_results.get("elements", []), Path(pdf_path).parent)
    # print(f"HTML 저장 완료: {html_path}")

    # 3) Summarizer 생성 후 문서 요약 실행
    self = IncrementalPdfSummarizer(verbose = True)
    summarized_results = self.summarize_document(parsed_results)