# pdf_parser_v3.py
import pymupdf
import os
import re
import requests
import json
from typing import TypedDict
from abc import ABC, abstractmethod
from PIL import Image
import base64
import pprint
import pickle
import xml.etree.ElementTree as ET


from dotenv import load_dotenv
load_dotenv(dotenv_path='./.env')
das
#########################################################
# 1. 노드 정의 ###########################################
#########################################################
# 문서 분할
class GraphState(TypedDict):
    filepath: str  # path
    filetype: str  # pdf
    page_numbers: list[int]  # page numbers
    batch_size: int  # batch size
    split_filepaths: list[str]  # split files
    analyzed_files: list[str]  # analyzed files
    page_elements: dict[int, dict[str, list[dict]]]  # page elements
    page_metadata: dict[int, dict]  # page metadata
    page_summary: dict[int, str]  # page summary
    images: list[str]  # image paths
    image_summary: list[str]  # image summary
    tables: list[str]  # table
    table_summary: dict[int, str]  # table summary
    table_markdown: dict[int, str]  # table markdown
    texts: list[str]  # text
    text_summary: dict[int, str]  # text summary
    table_summary_data_batches: list[dict]  # table summary data batches
    language: str  # language
class BaseNode(ABC):
    def __init__(self, verbose=False, **kwargs):
        self.name = self.__class__.__name__
        self.verbose = verbose

    @abstractmethod
    def execute(self, state: GraphState) -> GraphState:
        pass

    def log(self, message: str, **kwargs):
        if self.verbose:
            print(f"[{self.name}] {message}")
            for key, value in kwargs.items():
                print(f"  {key}: {value}")

    def __call__(self, state: GraphState) -> GraphState:
        return self.execute(state)
class SplitPDFFilesNode(BaseNode):

    def __init__(self, batch_size=10, **kwargs):
        super().__init__(**kwargs)
        self.name = "SplitPDFNode"
        self.batch_size = batch_size

    def execute(self, state: GraphState) -> GraphState:
        """
        입력 PDF를 여러 개의 작은 PDF 파일로 분할합니다.

        :param state: GraphState 객체, PDF 파일 경로와 배치 크기 정보를 포함
        :return: 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체
        """
        # PDF 파일 경로와 배치 크기 추출
        filepath = state["filepath"]

        # PDF 파일 열기
        input_pdf = pymupdf.open(filepath)
        num_pages = len(input_pdf)
        print(f"총 페이지 수: {num_pages}")

        ret = []
        # PDF 분할 작업 시작
        for start_page in range(0, num_pages, self.batch_size):
            # 배치의 마지막 페이지 계산 (전체 페이지 수를 초과하지 않도록)
            end_page = min(start_page + self.batch_size, num_pages) - 1

            # 분할된 PDF 파일명 생성
            input_file_basename = os.path.splitext(filepath)[0]
            output_file = f"{input_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
            print(f"분할 PDF 생성: {output_file}")

            # 새로운 PDF 파일 생성 및 페이지 삽입
            with pymupdf.open() as output_pdf:
                output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
                output_pdf.save(output_file)
                ret.append(output_file)

        # 원본 PDF 파일 닫기
        input_pdf.close()

        # 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체 반환
        return GraphState(split_filepaths=ret)
split_pdf_node = SplitPDFFilesNode(batch_size=10)

# Layout Analyzer
class LayoutAnalyzer:
    def __init__(self, api_key):
        """
        LayoutAnalyzer 클래스의 생성자

        :param api_key: Upstage API 인증을 위한 API 키
        """
        self.api_key = api_key

    def _upstage_layout_analysis(self, input_file):
        """
        Upstage의 레이아웃 분석 API를 호출하여 문서 분석을 수행합니다.

        :param input_file: 분석할 PDF 파일의 경로
        :return: 분석 결과가 저장된 JSON 파일의 경로
        """
        output_file = os.path.splitext(input_file)[0] + ".json"
        if os.path.exists(output_file):
            # 이미 존재하면 재활용
            print(f"이미 분석된 파일입니다. {output_file} 를 재활용합니다.")
            return output_file

        # API 요청 헤더 설정
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # API 요청 데이터 설정 (OCR 비활성화)
        data = {"ocr": False}

        # 분석할 PDF 파일 열기
        files = {"document": open(input_file, "rb")}

        # API 요청 보내기
        response = requests.post(
            "https://api.upstage.ai/v1/document-ai/layout-analysis",
            headers=headers,
            data=data,
            files=files,
        )

        # API 응답 처리 및 결과 저장
        if response.status_code == 200:
            # 분석 결과를 저장할 JSON 파일 경로 생성
            output_file = os.path.splitext(input_file)[0] + ".json"
            # 분석 결과를 JSON 파일로 저장
            with open(output_file, "w") as f:
                json.dump(response.json(), f, ensure_ascii=False)
            return output_file
        else:
            # API 요청이 실패한 경우 예외 발생
            raise ValueError(f"API 요청 실패. 상태 코드: {response.status_code}")

    def execute(self, input_file):
        """
        주어진 입력 파일에 대해 레이아웃 분석을 실행합니다.

        :param input_file: 분석할 PDF 파일의 경로
        :return: 분석 결과가 저장된 JSON 파일의 경로
        """
        return self._upstage_layout_analysis(input_file)
class LayoutAnalyzerNode(BaseNode):

    def __init__(self, api_key, **kwargs):
        super().__init__(**kwargs)
        self.name = "LayoutAnalyzerNode"
        self.api_key = api_key
        self.layout_analyzer = LayoutAnalyzer(api_key)

    def execute(self, state: GraphState) -> GraphState:
        # 분할된 PDF 파일 목록을 가져옵니다.
        split_files = state["split_filepaths"]

        # LayoutAnalyzer 객체를 생성합니다. API 키는 환경 변수에서 가져옵니다.
        analyzer = LayoutAnalyzer(self.api_key)

        # 분석된 파일들의 경로를 저장할 리스트를 초기화합니다.
        analyzed_files = []

        # 각 분할된 PDF 파일에 대해 레이아웃 분석을 수행합니다.
        for file in split_files:
            # 레이아웃 분석을 실행하고 결과 파일 경로를 리스트에 추가합니다.
            analyzed_files.append(analyzer.execute(file))

        # 분석된 파일 경로들을 정렬하여 새로운 GraphState 객체를 생성하고 반환합니다.
        # 정렬은 파일들의 순서를 유지하기 위해 수행됩니다.
        return GraphState(analyzed_files=sorted(analyzed_files))
layout_analyze_node = LayoutAnalyzerNode(os.environ.get("UPSTAGE_API_KEY"))

# 페이지 요소 추출
class ExtractPageElementsNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ExtractPageElementsNode"

    def extract_start_end_page(self, filename):
        """
        파일 이름에서 시작 페이지와 끝 페이지 번호를 추출하는 함수입니다.

        :param filename: 분석할 파일의 이름
        :return: 시작 페이지 번호와 끝 페이지 번호를 튜플로 반환
        """
        file_name = os.path.basename(filename)
        file_name_parts = file_name.split("_")

        if len(file_name_parts) >= 3:
            start_page = int(re.findall(r"(\d+)", file_name_parts[-2])[0])
            end_page = int(re.findall(r"(\d+)", file_name_parts[-1])[0])
        else:
            start_page, end_page = 0, 0

        return start_page, end_page

    def execute(self, state: GraphState) -> GraphState:
        """
        분석된 JSON 파일들에서 페이지 메타데이터를 추출하고 페이지 요소를 추출하는 함수입니다.

        :param state: 현재의 GraphState 객체
        :return: 페이지 메타데이터, 페이지 요소, 페이지 번호가 추가된 새로운 GraphState 객체
        """
        json_files = state["analyzed_files"]
        page_metadata = dict()
        page_elements = dict()
        element_id = 0

        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)

            start_page, _ = self.extract_start_end_page(json_file)

            for element in data["metadata"]["pages"]:
                original_page = int(element["page"])
                relative_page = start_page + original_page - 1

                metadata = {
                    "size": [
                        int(element["width"]),
                        int(element["height"]),
                    ],
                }
                page_metadata[relative_page] = metadata

            for element in data["elements"]:
                original_page = int(element["page"])
                relative_page = start_page + original_page - 1

                if relative_page not in page_elements:
                    page_elements[relative_page] = []

                element["id"] = element_id
                element_id += 1

                element["page"] = relative_page
                page_elements[relative_page].append(element)

        parsed_page_elements = self.extract_tag_elements_per_page(page_elements)
        page_numbers = list(parsed_page_elements.keys())
        return GraphState(
            page_metadata=page_metadata,
            page_elements=parsed_page_elements,
            page_numbers=page_numbers,
        )

    def extract_tag_elements_per_page(self, page_elements):
        # 파싱된 페이지 요소들을 저장할 새로운 딕셔너리를 생성합니다.
        parsed_page_elements = dict()

        # 각 페이지와 해당 페이지의 요소들을 순회합니다.
        for key, page_element in page_elements.items():
            # 이미지, 테이블, 텍스트 요소들을 저장할 리스트를 초기화합니다.
            image_elements = []
            table_elements = []
            text_elements = []

            # 페이지의 각 요소를 순회하며 카테고리별로 분류합니다.
            for element in page_element:
                if element["category"] == "figure":
                    # 이미지 요소인 경우 image_elements 리스트에 추가합니다.
                    image_elements.append(element)
                elif element["category"] == "table":
                    # 테이블 요소인 경우 table_elements 리스트에 추가합니다.
                    table_elements.append(element)
                else:
                    # 그 외의 요소는 모두 텍스트 요소로 간주하여 text_elements 리스트에 추가합니다.
                    text_elements.append(element)

            # 분류된 요소들을 페이지 키와 함께 새로운 딕셔너리에 저장합니다.
            parsed_page_elements[key] = {
                "image_elements": image_elements,
                "table_elements": table_elements,
                "text_elements": text_elements,
                "elements": page_element,  # 원본 페이지 요소도 함께 저장합니다.
            }

        return parsed_page_elements
page_element_extractor_node = ExtractPageElementsNode()

# 이미지 자르기
class ImageCropper:
    @staticmethod
    def pdf_to_image(pdf_file, page_num, dpi=300):
        """
        PDF 파일의 특정 페이지를 이미지로 변환하는 메서드

        :param page_num: 변환할 페이지 번호 (1부터 시작)
        :param dpi: 이미지 해상도 (기본값: 300)
        :return: 변환된 이미지 객체
        """
        with pymupdf.open(pdf_file) as doc:
            page = doc[page_num].get_pixmap(dpi=dpi)
            target_page_size = [page.width, page.height]
            page_img = Image.frombytes("RGB", target_page_size, page.samples)
        return page_img

    @staticmethod
    def normalize_coordinates(coordinates, output_page_size):
        """
        좌표를 정규화하는 정적 메서드

        :param coordinates: 원본 좌표 리스트
        :param output_page_size: 출력 페이지 크기 [너비, 높이]
        :return: 정규화된 좌표 (x1, y1, x2, y2)
        """
        x_values = [coord["x"] for coord in coordinates]
        y_values = [coord["y"] for coord in coordinates]
        x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)

        return (
            x1 / output_page_size[0],
            y1 / output_page_size[1],
            x2 / output_page_size[0],
            y2 / output_page_size[1],
        )

    @staticmethod
    def crop_image(img, coordinates, output_file):
        """
        이미지를 주어진 좌표에 따라 자르고 저장하는 정적 메서드

        :param img: 원본 이미지 객체
        :param coordinates: 정규화된 좌표 (x1, y1, x2, y2)
        :param output_file: 저장할 파일 경로
        """
        img_width, img_height = img.size
        x1, y1, x2, y2 = [
            int(coord * dim)
            for coord, dim in zip(coordinates, [img_width, img_height] * 2)
        ]
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img.save(output_file)
class ImageCropperNode(BaseNode):
    """
    PDF 파일에서 이미지를 추출하고 크롭하는 노드
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ImageCropperNode"

    def execute(self, state: GraphState) -> GraphState:
        """
        PDF 파일에서 이미지를 추출하고 크롭하는 함수

        :param state: GraphState 객체
        :return: 크롭된 이미지 정보가 포함된 GraphState 객체
        """
        pdf_file = state["filepath"]  # PDF 파일 경로
        page_numbers = state["page_numbers"]  # 처리할 페이지 번호 목록
        output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
        os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

        cropped_images = dict()  # 크롭된 이미지 정보를 저장할 딕셔너리
        for page_num in page_numbers:
            pdf_image = ImageCropper.pdf_to_image(
                pdf_file, page_num
            )  # PDF 페이지를 이미지로 변환
            for element in state["page_elements"][page_num]["image_elements"]:
                if element["category"] == "figure":
                    # 이미지 요소의 좌표를 정규화
                    normalized_coordinates = ImageCropper.normalize_coordinates(
                        element["bounding_box"],
                        state["page_metadata"][page_num]["size"],
                    )

                    # 크롭된 이미지 저장 경로 설정
                    output_file = os.path.join(output_folder, f"{element['id']}.png")
                    # 이미지 크롭 및 저장
                    ImageCropper.crop_image(
                        pdf_image, normalized_coordinates, output_file
                    )
                    cropped_images[element["id"]] = output_file
                    print(f"page:{page_num}, id:{element['id']}, path: {output_file}")
        return GraphState(
            images=cropped_images
        )  # 크롭된 이미지 정보를 포함한 GraphState 반환
image_cropper_node = ImageCropperNode()

# 테이블 자르기
class TableCropperNode(BaseNode):
    """
    Table 이미지를 추출하고 크롭하는 노드
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TableCropperNode"

    def execute(self, state: GraphState) -> GraphState:
        """
        PDF 파일에서 표를 추출하고 크롭하는 함수

        :param state: GraphState 객체
        :return: 크롭된 표 이미지 정보가 포함된 GraphState 객체
        """
        pdf_file = state["filepath"]  # PDF 파일 경로
        page_numbers = state["page_numbers"]  # 처리할 페이지 번호 목록
        output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
        os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

        cropped_images = dict()  # 크롭된 표 이미지 정보를 저장할 딕셔너리
        for page_num in page_numbers:
            pdf_image = ImageCropper.pdf_to_image(
                pdf_file, page_num
            )  # PDF 페이지를 이미지로 변환
            for element in state["page_elements"][page_num]["table_elements"]:
                if element["category"] == "table":
                    # 표 요소의 좌표를 정규화
                    normalized_coordinates = ImageCropper.normalize_coordinates(
                        element["bounding_box"],
                        state["page_metadata"][page_num]["size"],
                    )

                    # 크롭된 표 이미지 저장 경로 설정
                    output_file = os.path.join(output_folder, f"{element['id']}.png")
                    # 표 이미지 크롭 및 저장
                    ImageCropper.crop_image(
                        pdf_image, normalized_coordinates, output_file
                    )
                    cropped_images[element["id"]] = output_file
                    print(f"page:{page_num}, id:{element['id']}, path: {output_file}")
        return GraphState(
            tables=cropped_images
        )  # 크롭된 표 이미지 정보를 포함한 GraphState 반환
table_cropper_node = TableCropperNode()

# 페이지별 텍스트 추출
class ExtractPageTextNode(BaseNode):
    """
    페이지별 텍스트를 추출하는 노드
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ExtractPageTextNode"

    def execute(self, state: GraphState) -> GraphState:
        # 상태 객체에서 페이지 번호 목록을 가져옵니다.
        page_numbers = state["page_numbers"]

        # 추출된 텍스트를 저장할 딕셔너리를 초기화합니다.
        extracted_texts = dict()

        # 각 페이지 번호에 대해 반복합니다.
        for page_num in page_numbers:
            # 현재 페이지의 텍스트를 저장할 빈 문자열을 초기화합니다.
            extracted_texts[page_num] = ""

            # 현재 페이지의 모든 텍스트 요소에 대해 반복합니다.
            for element in state["page_elements"][page_num]["text_elements"]:
                # 각 텍스트 요소의 내용을 현재 페이지의 텍스트에 추가합니다.
                extracted_texts[page_num] += element["text"]

        # 추출된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
        return GraphState(texts=extracted_texts)
extract_page_text = ExtractPageTextNode()

# 페이지별 요약
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
class CreatePageSummaryNode(BaseNode):
    """
    페이지별 요약을 생성하는 노드
    """

    def __init__(self, api_key, **kwargs):
        super().__init__(**kwargs)
        self.name = "CreatePageSummaryNode"
        self.api_key = api_key

    def create_text_summary_chain(self):
        # 요약을 위한 프롬프트 템플릿을 정의합니다.
        prompt = PromptTemplate.from_template(
            """Please summarize the sentence according to the following REQUEST.

        REQUEST:
        1. Summarize the main points in bullet points.
        2. Write the summary in same language as the context.
        3. DO NOT translate any technical terms.
        4. DO NOT include any unnecessary information.
        5. Summary must include important entities, numerical values.

        CONTEXT:
        {context}

        SUMMARY:"
        """
        )

        # ChatOpenAI 모델의 또 다른 인스턴스를 생성합니다. (이전 인스턴스와 동일한 설정)
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            api_key=self.api_key,
        )

        # 문서 요약을 위한 체인을 생성합니다.
        # 이 체인은 여러 문서를 입력받아 하나의 요약된 텍스트로 결합합니다.
        text_summary_chain = create_stuff_documents_chain(llm, prompt)

        return text_summary_chain

    def execute(self, state: GraphState) -> GraphState:
        # state에서 텍스트 데이터를 가져옵니다.
        texts = state["texts"]

        # 요약된 텍스트를 저장할 딕셔너리를 초기화합니다.
        text_summary = dict()

        # texts.items()를 페이지 번호(키)를 기준으로 오름차순 정렬합니다.
        sorted_texts = sorted(texts.items(), key=lambda x: x[0])

        # 각 페이지의 텍스트를 Document 객체로 변환하여 입력 리스트를 생성합니다.
        inputs = [
            {"context": [Document(page_content=text)]}
            for page_num, text in sorted_texts
        ]
        # 요약 체인 생성
        text_summary_chain = self.create_text_summary_chain()

        # text_summary_chain을 사용하여 일괄 처리로 요약을 생성합니다.
        summaries = text_summary_chain.batch(inputs)

        # 생성된 요약을 페이지 번호와 함께 딕셔너리에 저장합니다.
        for page_num, summary in enumerate(summaries):
            text_summary[page_num] = summary

        # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
        return GraphState(text_summary=text_summary)
page_summary_node = CreatePageSummaryNode(api_key=os.environ.get("OPENAI_API_KEY"))

# 이미지 요약
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
class MultiModal:
    def __init__(self, model, system_prompt=None, user_prompt=None):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.init_prompt()

    def init_prompt(self):
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful assistant who helps users to write a report related to images in Korean."
        if self.user_prompt is None:
            self.user_prompt = "Explain the images as an alternative text in Korean."

    # 이미지를 base64로 인코딩하는 함수 (URL)
    def encode_image_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            image_content = response.content
            if url.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif url.lower().endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/unknown"
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"
        else:
            raise Exception("Failed to download image")

    # 이미지를 base64로 인코딩하는 함수 (파일)
    def encode_image_from_file(self, file_path):
        with open(file_path, "rb") as image_file:
            image_content = image_file.read()
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif file_ext == ".png":
                mime_type = "image/png"
            else:
                mime_type = "image/unknown"
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"

    # 이미지 경로에 따라 적절한 함수를 호출하는 함수
    def encode_image(self, image_path):
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return self.encode_image_from_url(image_path)
        else:
            return self.encode_image_from_file(image_path)

    def display_image(self, encoded_image):
        display(Image(url=encoded_image))

    def create_messages(
            self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        encoded_image = self.encode_image(image_url)
        if display_image:
            self.display_image(encoded_image)

        system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        user_prompt = user_prompt if user_prompt is not None else self.user_prompt

        # 인코딩된 이미지를 사용하여 다른 처리를 수행할 수 있습니다.
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{encoded_image}"},
                    },
                ],
            },
        ]
        return messages

    def invoke(
            self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        response = self.model.invoke(messages)
        return response.content

    def batch(
            self,
            image_urls: list[str],
            system_prompts: list[str] = [],
            user_prompts: list[str] = [],
            display_image=False,
    ):
        messages = []
        for image_url, system_prompt, user_prompt in zip(
                image_urls, system_prompts, user_prompts
        ):
            message = self.create_messages(
                image_url, system_prompt, user_prompt, display_image
            )
            messages.append(message)
        response = self.model.batch(messages)
        return [r.content for r in response]

    def stream(
            self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        response = self.model.stream(messages)
        return response
@chain
def extract_image_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = """You are an expert in extracting useful information from IMAGE.
With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval.
Also, provide five hypothetical questions based on the image that users can ask.
"""

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["image"]
        language = data_batch["language"]
        user_prompt_template = f"""Here is the context related to the image: {context}

###

Output Format:

<image>
<title>
[title]
</title>
<summary>
[summary]
</summary>
<entities> 
[entities]
</entities>
<hypothetical_questions>
[hypothetical_questions]
</hypothetical_questions>
</image>

Output must be written in {language}.
"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer
class CreateImageSummaryNode(BaseNode):
    """
    이미지 요약을 생성하는 노드
    """

    def __init__(self, api_key, **kwargs):
        super().__init__(**kwargs)
        self.name = "CreateImageSummaryNode"
        self.api_key = api_key

    def create_image_summary_data_batches(self, state: GraphState):
        # 이미지 요약을 위한 데이터 배치를 생성하는 함수
        data_batches = []

        # 페이지 번호를 오름차순으로 정렬
        page_numbers = sorted(list(state["page_elements"].keys()))

        for page_num in page_numbers:
            # 각 페이지의 요약된 텍스트를 가져옴
            text = state["text_summary"][page_num]
            # 해당 페이지의 모든 이미지 요소에 대해 반복
            for image_element in state["page_elements"][page_num]["image_elements"]:
                # 이미지 ID를 정수로 변환
                image_id = int(image_element["id"])

                # 데이터 배치에 이미지 정보, 관련 텍스트, 페이지 번호, ID를 추가
                data_batches.append(
                    {
                        "image": state["images"][image_id],  # 이미지 파일 경로
                        "text": text,  # 관련 텍스트 요약
                        "page": page_num,  # 페이지 번호
                        "id": image_id,  # 이미지 ID
                        "language": state["language"],  # 언어
                    }
                )
        # 생성된 데이터 배치를 GraphState 객체에 담아 반환
        return data_batches

    def execute(self, state: GraphState):
        image_summary_data_batches = self.create_image_summary_data_batches(state)
        # 이미지 요약 추출
        # extract_image_summary 함수를 호출하여 이미지 요약 생성
        image_summaries = extract_image_summary.invoke(
            image_summary_data_batches,
        )

        # 이미지 요약 결과를 저장할 딕셔너리 초기화
        image_summary_output = dict()

        # 각 데이터 배치와 이미지 요약을 순회하며 처리
        for data_batch, image_summary in zip(
            image_summary_data_batches, image_summaries
        ):
            # 데이터 배치의 ID를 키로 사용하여 이미지 요약 저장
            image_summary_output[data_batch["id"]] = image_summary

        # 이미지 요약 결과를 포함한 새로운 GraphState 객체 반환
        return GraphState(image_summary=image_summary_output)
image_summary_node = CreateImageSummaryNode(api_key=os.environ.get("OPENAI_API_KEY"))

# 테이블 요약
@chain
def extract_table_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = """You are an expert in extracting useful information from TABLE. 
With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval.
If the numbers are present, summarize important insights from the numbers.
Also, provide five hypothetical questions based on the image that users can ask.
"""

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["table"]
        language = data_batch["language"]
        user_prompt_template = f"""Here is the context related to the image of table: {context}

###

Output Format:

<table>
<title>
[title]
</title>
<summary>
[summary]
</summary>
<entities> 
[entities]
</entities>
<data_insights>
[data_insights]
</data_insights>
<hypothetical_questions>
[hypothetical_questions]
</hypothetical_questions>
</table>

Output must be written in {language}.
"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer
class CreateTableSummaryNode(BaseNode):
    """
    테이블 요약을 생성하는 노드
    """

    def __init__(self, api_key, **kwargs):
        super().__init__(**kwargs)
        self.name = "CreateTableSummaryNode"
        self.api_key = api_key

    def create_table_summary_data_batches(self, state: GraphState):
        # 테이블 요약을 위한 데이터 배치를 생성하는 함수
        data_batches = []

        # 페이지 번호를 오름차순으로 정렬
        page_numbers = sorted(list(state["page_elements"].keys()))

        for page_num in page_numbers:
            # 각 페이지의 요약된 텍스트를 가져옴
            text = state["text_summary"][page_num]
            # 해당 페이지의 모든 테이블 요소에 대해 반복
            for image_element in state["page_elements"][page_num]["table_elements"]:
                # 테이블 ID를 정수로 변환
                image_id = int(image_element["id"])

                # 데이터 배치에 테이블 정보, 관련 텍스트, 페이지 번호, ID를 추가
                data_batches.append(
                    {
                        "table": state["tables"][image_id],  # 테이블 데이터
                        "text": text,  # 관련 텍스트 요약
                        "page": page_num,  # 페이지 번호
                        "id": image_id,  # 테이블 ID
                        "language": state["language"],  # 언어
                    }
                )
        # 생성된 데이터 배치를 GraphState 객체에 담아 반환
        # return GraphState(table_summary_data_batches=data_batches)
        return data_batches

    def execute(self, state: GraphState):
        table_summary_data_batches = self.create_table_summary_data_batches(state)
        # 테이블 요약 추출
        table_summaries = extract_table_summary.invoke(
            table_summary_data_batches,
        )

        # 테이블 요약 결과를 저장할 딕셔너리 초기화
        table_summary_output = dict()

        # 각 데이터 배치와 테이블 요약을 순회하며 처리
        for data_batch, table_summary in zip(
            table_summary_data_batches, table_summaries
        ):
            # 데이터 배치의 ID를 키로 사용하여 테이블 요약 저장
            table_summary_output[data_batch["id"]] = table_summary

        # 테이블 요약 결과를 포함한 새로운 GraphState 객체 반환
        return GraphState(
            table_summary=table_summary_output,
            table_summary_data_batches=table_summary_data_batches,
        )
table_summary_node = CreateTableSummaryNode(api_key=os.environ.get("OPENAI_API_KEY"))

# 테이블 Markdown 추출
class TableMarkdownExtractorNode(BaseNode):
    """
    테이블 이미지를 마크다운 테이블로 변환하는 노드
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TableMarkdownExtractorNode"

    def execute(self, state: GraphState):
        # table_markdown_extractor를 사용하여 테이블 마크다운 생성
        # state["table_summary_data_batches"]에 저장된 테이블 데이터를 사용
        table_markdowns = table_markdown_extractor.invoke(
            state["table_summary_data_batches"],
        )

        # 결과를 저장할 딕셔너리 초기화
        table_markdown_output = dict()

        # 각 데이터 배치와 생성된 테이블 마크다운을 매칭하여 저장
        for data_batch, table_summary in zip(
            state["table_summary_data_batches"], table_markdowns
        ):
            # 데이터 배치의 id를 키로 사용하여 테이블 마크다운 저장
            table_markdown_output[data_batch["id"]] = table_summary

        # 새로운 GraphState 객체 반환, table_markdown 키에 결과 저장
        return GraphState(table_markdown=table_markdown_output)
table_markdown_extractor = TableMarkdownExtractorNode()



#########################################################
# 2. 그래프 정의 ###########################################
#########################################################
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
# LangGraph을 생성
workflow = StateGraph(GraphState)

# 노드들을 정의합니다.
workflow.add_node("split_pdf_node", split_pdf_node)
workflow.add_node("layout_analyzer_node", layout_analyze_node)
workflow.add_node("page_element_extractor_node", page_element_extractor_node)
workflow.add_node("image_cropper_node", image_cropper_node)
workflow.add_node("table_cropper_node", table_cropper_node)
workflow.add_node("extract_page_text_node", extract_page_text)
workflow.add_node("page_summary_node", page_summary_node)
workflow.add_node("image_summary_node", image_summary_node)
workflow.add_node("table_summary_node", table_summary_node)
workflow.add_node("table_markdown_node", table_markdown_extractor)

# 각 노드들을 연결합니다.
workflow.add_edge("split_pdf_node", "layout_analyzer_node")
workflow.add_edge("layout_analyzer_node", "page_element_extractor_node")
workflow.add_edge("page_element_extractor_node", "image_cropper_node")
workflow.add_edge("page_element_extractor_node", "table_cropper_node")
workflow.add_edge("page_element_extractor_node", "extract_page_text_node")
workflow.add_edge("image_cropper_node", "page_summary_node")
workflow.add_edge("table_cropper_node", "page_summary_node")
workflow.add_edge("extract_page_text_node", "page_summary_node")
workflow.add_edge("page_summary_node", "image_summary_node")
workflow.add_edge("page_summary_node", "table_summary_node")
workflow.add_edge("image_summary_node", END)
workflow.add_edge("table_summary_node", "table_markdown_node")
workflow.add_edge("table_markdown_node", END)

workflow.set_entry_point("split_pdf_node")

memory = MemorySaver()
Visualize_Graph=True
app = workflow.compile(checkpointer=memory)
if Visualize_Graph:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))




from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError


import glob
pdf_file_list  = [x.replace('\\', '/') for x in glob.glob('./../data/*.pdf')]
LANGUAGE = "Korean"

config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "Test-Parser"})

# AgentState 객체를 활용하여 질문을 입력합니다.
# FILEPATH = './../data/신한투자증권_산업_반도체 및 관련장비_20250516181226.pdf'
all_docs = []
for FILEPATH in pdf_file_list:
    inputs = GraphState(filepath=FILEPATH, language=LANGUAGE)

    # app.stream을 통해 입력된 메시지에 대한 출력을 스트리밍합니다.
    try:
        for output in app.stream(inputs, config=config):
            # 출력된 결과에서 키와 값을 순회합니다.
            for key, value in output.items():
                # 노드의 이름과 해당 노드에서 나온 출력을 출력합니다.
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint("---")
                # 출력 값을 예쁘게 출력합니다.
                pprint.pprint(value, indent=2, width=80, depth=None)
            # 각 출력 사이에 구분선을 추가합니다.
            pprint.pprint("\n---\n")
    except GraphRecursionError as e:
        pprint.pprint(f"Recursion limit reached: {e}")

    state = app.get_state(config).values

    #########################################################
    # 3. 저장 & 불러오기 ######################################
    #########################################################
    def save_state(state, filepath):
        """상태를 pickle 파일로 저장합니다."""
        base, _ = os.path.splitext(filepath)
        with open(f"{base}.pkl", "wb") as f:
            pickle.dump(state, f)
    def load_state(filepath):
        """pickle 파일에서 상태를 불러옵니다."""
        base, _ = os.path.splitext(filepath)
        with open(f"{base}.pkl", "rb") as f:
            return pickle.load(f)

    # 상태 저장
    save_state(state, FILEPATH)

    # 상태 불러오기
    loaded_state = load_state(FILEPATH)
    print(loaded_state)



    state = loaded_state


    #########################################################
    # 4. Markdown 생성 ######################################
    #########################################################
    ### Image, Table 에서 추출된 데이터 Vector DB 생성을 위한 문서 생성
    # - Title, Summary, Entities 는 임베딩 검색에 걸리기 위한 문서로 생성
    # - hypothetical_questions 는 임베딩 검색에 걸리기 위한 문서로 생성
    def convert_to_markdown_table(table_summary):
        html = "<table>\n"

        # table_summary가 문자열인 경우를 처리합니다
        if isinstance(table_summary, str):
            # XML 파싱을 사용하여 문자열에서 데이터를 추출합니다
            root = ET.fromstring(table_summary)
            for child in root:
                html += f"  <tr>\n    <th>{child.tag}</th>\n    <td>"

                if child.tag in ["entities", "data_insights"]:
                    html += "<ul>\n"
                    for item in child.text.strip().split("\n- "):
                        if item.strip():
                            html += f"      <li>{item.strip()}</li>\n"
                    html += "    </ul>"
                elif child.tag == "hypothetical_questions":
                    html += "<ol>\n"
                    for item in child.text.strip().split("\n"):
                        if item.strip():
                            html += f"      <li>{item.strip()}</li>\n"
                    html += "    </ol>"
                else:
                    html += child.text.strip()

                html += "</td>\n  </tr>\n"
        else:
            # 기존의 딕셔너리 처리 로직을 유지합니다
            for key, value in table_summary.items():
                html += f"  <tr>\n    <th>{key}</th>\n    <td>"

                if key in ["entities", "data_insights"]:
                    html += "<ul>\n"
                    for item in value.split("\n- "):
                        if item.strip():
                            html += f"      <li>{item.strip()}</li>\n"
                    html += "    </ul>"
                elif key == "hypothetical_questions":
                    html += "<ol>\n"
                    for item in value.split("\n"):
                        if item.strip():
                            html += f"      <li>{item.strip()}</li>\n"
                    html += "    </ol>"
                else:
                    html += value

                html += "</td>\n  </tr>\n"

        html += "</table>"
        return html
    def extract_tag_content(content, tag):
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            return None
    def extract_non_tag_content(content, tag):
        pattern = rf"<{tag}>.*?</{tag}>"
        result = re.sub(pattern, "", content, flags=re.DOTALL)
        return result.strip()

    def create_document(content, metadata):
        """
        문서 객체를 생성합니다.

        Args:
            content (str): 문서의 내용
            metadata (dict): 문서의 메타데이터

        Returns:
            Document: 생성된 문서 객체
        """
        return Document(page_content=content, metadata=metadata)
    def process_text_element(element):
        """
        텍스트 요소를 처리합니다.

        Args:
            element (dict): 텍스트 요소 정보

        Returns:
            str: 텍스트 내용
        """
        return element["text"]
    def process_image_element(element, state, page_number):
        """
        이미지 요소를 처리합니다.

        Args:
            element (dict): 이미지 요소 정보
            state (dict): 현재 상태
            page_number (str): 페이지 번호

        Returns:
            tuple: 마크다운 문자열과 문서 객체 리스트
        """
        image_id = str(element["id"])
        image_summary = state["image_summary"][image_id]
        image_path = state["images"][image_id]
        image_path_md = f"![{image_path}]({image_path})"

        image_summary_md = convert_to_markdown_table(image_summary)
        markdown = f"{image_path_md}"

        image_summary_clean = extract_non_tag_content(
            image_summary, "hypothetical_questions"
        )

        docs = [
            create_document(
                image_summary_clean,
                {
                    "type": "image",
                    "image": image_path,
                    "page": page_number,
                    "source": state["filepath"],
                    "id": image_id,
                },
            )
        ]

        hypo_docs = []

        hypothetical_questions = extract_tag_content(
            image_summary, "hypothetical_questions"
        )
        if hypothetical_questions != None:
            hypo_docs.append(
                create_document(
                    hypothetical_questions,
                    {
                        "type": "hypothetical_questions",
                        "image": image_path,
                        "summary": image_summary_clean,
                        "page": page_number,
                        "source": state["filepath"],
                        "id": image_id,
                    },
                )
            )

        return markdown, docs, hypo_docs
    def process_table_element(element, state, page_number):
        """
        테이블 요소를 처리합니다.

        Args:
            element (dict): 테이블 요소 정보
            state (dict): 현재 상태
            page_number (str): 페이지 번호

        Returns:
            tuple: 마크다운 문자열과 문서 객체
        """
        table_id = str(element["id"])
        table_summary = state["table_summary"][table_id]
        table_markdown = state["table_markdown"][table_id]
        table_path = state["tables"][table_id]
        table_path_md = f"![{table_path}]({table_path})"

        table_summary_md = convert_to_markdown_table(table_summary)
        markdown = f"{table_path_md}\n{table_markdown}"

        table_summary_clean = extract_non_tag_content(
            table_summary, "hypothetical_questions"
        )

        docs = [
            create_document(
                table_summary_clean,
                {
                    "type": "table",
                    "table": table_path,
                    "markdown": table_markdown,
                    "page": page_number,
                    "source": state["filepath"],
                    "id": table_id,
                },
            )
        ]

        hypo_docs = []

        hypothetical_questions = extract_tag_content(
            table_summary, "hypothetical_questions"
        )
        if hypothetical_questions != None:
            hypo_docs.append(
                create_document(
                    hypothetical_questions,
                    {
                        "type": "hypothetical_questions",
                        "table": table_path,
                        "summary": table_summary_clean,
                        "markdown": table_markdown,
                        "page": page_number,
                        "source": state["filepath"],
                        "id": table_id,
                    },
                )
            )

        return markdown, docs, hypo_docs
    def process_page(page, state, page_number, text_splitter):
        """
        페이지를 처리합니다.

        Args:
            page (dict): 페이지 정보
            state (dict): 현재 상태
            page_number (str): 페이지 번호
            text_splitter (RecursiveCharacterTextSplitter): 텍스트 분할기

        Returns:
            tuple: 마크다운 문자열 리스트와 문서 객체 리스트
        """
        markdowns = []
        docs = []
        hypo_docs = []
        page_texts = []

        for element in page["elements"]:
            if element["category"] == "figure":
                markdown, element_docs, hypo_doc = process_image_element(
                    element, state, page_number
                )
                markdowns.append(markdown)
                docs.extend(element_docs)
                hypo_docs.extend(hypo_doc)
            elif element["category"] == "table":
                markdown, element_docs, hypo_doc = process_table_element(
                    element, state, page_number
                )
                markdowns.append(markdown)
                docs.extend(element_docs)
                hypo_docs.extend(hypo_doc)
            else:
                text = process_text_element(element)
                markdowns.append(text)
                page_texts.append(text)

        page_text = "\n".join(page_texts)
        split_texts = text_splitter.split_text(page_text)

        text_summary = state["text_summary"][str(page_number)]
        docs.append(
            create_document(
                text_summary,
                metadata={
                    "type": "page_summary",
                    "page": page_number,
                    "source": state["filepath"],
                    "text": page_text,
                },
            )
        )

        for text in split_texts:
            docs.append(
                create_document(
                    text,
                    metadata={
                        "type": "text",
                        "page": page_number,
                        "source": state["filepath"],
                        "summary": text_summary,
                    },
                )
            )

        return markdowns, docs, hypo_docs

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    def process_document(state):
        """
        전체 문서를 처리합니다.

        Args:
            state (dict): 현재 상태

        Returns:
            tuple: 마크다운 문자열 리스트와 문서 객체 리스트
        """
        markdowns = []
        docs = []
        hypo_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for page_number, page in state["page_elements"].items():
            page_markdowns, page_docs, page_hypo_docs = process_page(
                page, state, page_number, text_splitter
            )
            markdowns.extend(page_markdowns)
            docs.extend(page_docs)
            hypo_docs.extend(page_hypo_docs)

        return markdowns, docs, hypo_docs
    markdowns, docs, hypo_docs = process_document(state)

    all_docs.extend(docs + hypo_docs)

    # Markdown 파일로 텍스트 저장
    with open(FILEPATH.replace(".pdf", ".md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(markdowns))
    print(f"텍스트가 '{FILEPATH.replace('.pdf', '.md')}' 파일로 저장되었습니다.")


    print(f'docs:\n {docs}\n')
    print(f'hypo_docs:\n {hypo_docs}\n')
    print(docs[12].__dict__)
    for i, d in enumerate(docs):
        print(i, d.metadata["type"])
    print(docs[3].__dict__)
    print(docs[20].__dict__)
    print(docs[23].__dict__)
    print(hypo_docs[0].__dict__)
    print(hypo_docs[2].__dict__)




#########################################################
# 5. RAG 예제 ###########################################
#########################################################
all_docs = docs + hypo_docs
len(all_docs)


from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# bm25 retriever와 faiss retriever를 초기화합니다.
bm25_retriever = BM25Retriever.from_documents(all_docs,)
bm25_retriever.k = 5  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.

embedding = OpenAIEmbeddings()  # OpenAI 임베딩을 사용합니다.

faiss_vectorstore = FAISS.from_documents(all_docs, embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

# 앙상블 retriever를 초기화합니다.
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.7, 0.3])


# Relevance Checker 로직을 활용한 중요 정보 필터링
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class GradeRetrievalQuestion(BaseModel):
    """A binary score to determine the relevance of the retrieved documents to the question."""

    score: str = Field(
        description="Whether the retrieved context is relevant to the question, 'yes' or 'no'"
    )
class GradeRetrievalAnswer(BaseModel):
    """A binary score to determine the relevance of the retrieved documents to the answer."""

    score: str = Field(
        description="Whether the retrieved context is relevant to the answer, 'yes' or 'no'"
    )
class OpenAIRelevanceGrader:
    """
    OpenAI 기반의 관련성 평가기 클래스입니다.

    이 클래스는 검색된 문서가 주어진 질문이나 답변과 얼마나 관련이 있는지 평가합니다.
    'retrieval-question' 또는 'retrieval-answer' 두 가지 모드로 작동할 수 있습니다.

    Attributes:
        llm: 사용할 언어 모델 인스턴스
        structured_llm_grader: 구조화된 출력을 생성하는 LLM 인스턴스
        grader_prompt: 평가에 사용될 프롬프트 템플릿

    Args:
        llm: 사용할 언어 모델 인스턴스
        target (str): 평가 대상 ('retrieval-question' 또는 'retrieval-answer')
    """

    def __init__(self, llm, target="retrieval-question"):
        """
        OpenAIRelevanceGrader 클래스의 초기화 메서드입니다.

        Args:
            llm: 사용할 언어 모델 인스턴스
            target (str): 평가 대상 ('retrieval-question' 또는 'retrieval-answer')

        Raises:
            ValueError: 유효하지 않은 target 값이 제공될 경우 발생
        """
        self.llm = llm

        if target == "retrieval-question":
            self.structured_llm_grader = llm.with_structured_output(
                GradeRetrievalQuestion
            )
        elif target == "retrieval-answer":
            self.structured_llm_grader = llm.with_structured_output(
                GradeRetrievalAnswer
            )
        else:
            raise ValueError(f"Invalid target: {target}")

        # 프롬프트
        target_variable = (
            "user question" if target == "retrieval-question" else "answer"
        )
        system = f"""You are a grader assessing relevance of a retrieved document to a {target_variable}. \n 
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            If the document contains keyword(s) or semantic meaning related to the {target_variable}, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to {target_variable}."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"Retrieved document: \n\n {{context}} \n\n {target_variable}: {{input}}",
                ),
            ]
        )
        self.grader_prompt = grade_prompt

    def create(self):
        """
        관련성 평가기를 생성하고 반환합니다.

        Returns:
            관련성 평가를 수행할 수 있는 체인 객체
        """

        retrieval_grader_oai = self.grader_prompt | self.structured_llm_grader
        return retrieval_grader_oai
groundedness_check = OpenAIRelevanceGrader(ChatOpenAI(model="gpt-4o-mini", temperature=0), target="retrieval-question").create()


# 함수 사용 예시
# 앙상블 리트리버를 사용하여 질문에 대한 문서 검색
retrieved_documents = ensemble_retriever.invoke("디지털 정부혁신의 주요 목표는 무엇인가요?")
print('retrieved_documents')
print(retrieved_documents)

# 검색된 문서를 정제하여 깨끗한 형태로 변환
def clean_retrieved_documents(retrieved_documents):
    clean_docs = []

    for doc in retrieved_documents:
        metadata = doc.metadata
        new_metadata = {}
        content = doc.page_content

        # 문서 타입이 'page_summary' 또는 'text'인 경우
        if metadata["type"] in ["page_summary", "text"]:
            # 페이지 번호와 소스 정보를 새 메타데이터에 추가
            if "page" in metadata:
                new_metadata["page"] = metadata["page"]
            if "source" in metadata:
                new_metadata["source"] = metadata["source"]
            # 'text' 타입인 경우 요약 정보도 추가
            if metadata["type"] == "text":
                # content += f'\n\n<summary>{metadata["summary"]}</summary>'
                new_metadata["summary"] = metadata["summary"]
            clean_docs.append(Document(page_content=content, metadata=new_metadata))

        # 문서 타입이 'image'인 경우
        elif metadata["type"] == "image":
            image_path = metadata["image"]
            # 페이지 번호와 소스 정보를 새 메타데이터에 추가
            if "page" in metadata:
                new_metadata["page"] = metadata["page"]
            if "source" in metadata:
                new_metadata["source"] = metadata["source"]
            # 내용을 마크다운 테이블 형식으로 변환
            content = convert_to_markdown_table(content)

            clean_docs.append(Document(page_content=content, metadata=new_metadata))

        # 문서 타입이 'table'인 경우
        elif metadata["type"] == "table":
            table_path = metadata["table"]
            table_markdown = metadata["markdown"]
            # 페이지 번호와 소스 정보를 새 메타데이터에 추가
            if "page" in metadata:
                new_metadata["page"] = metadata["page"]
            if "source" in metadata:
                new_metadata["source"] = metadata["source"]
            # 내용을 마크다운 테이블 형식으로 변환하고 원본 마크다운과 결합
            content = f"{convert_to_markdown_table(content)}\n\n{table_markdown}"

            clean_docs.append(Document(page_content=content, metadata=new_metadata))

        # 문서 타입이 'hypothetical_questions'인 경우
        elif metadata["type"] == "hypothetical_questions":
            # 내용을 요약 정보로 대체
            content = metadata["summary"]
            # 페이지 번호와 소스 정보를 새 메타데이터에 추가
            if "page" in metadata:
                new_metadata["page"] = metadata["page"]
            if "source" in metadata:
                new_metadata["source"] = metadata["source"]
            clean_docs.append(Document(page_content=content, metadata=new_metadata))

    return clean_docs
cleaned_documents = clean_retrieved_documents(retrieved_documents)
print('cleaned_documents')
print(cleaned_documents)

for doc in cleaned_documents:
    print(doc.page_content)
    print("---" * 30)
    print(doc.metadata)
    print("===" * 30, end="\n\n\n")

def retrieve_and_check(question, use_checker=True):
    # 질문에 대한 문서를 검색합니다.
    retrieved_documents = ensemble_retriever.invoke(question)

    # 검색된 문서를 정제합니다.
    cleaned_documents = clean_retrieved_documents(retrieved_documents)

    filtered_documents = []
    if use_checker:
        # 검사기를 사용하는 경우, 각 문서의 내용과 질문을 입력으로 준비합니다.
        checking_inputs = [
            {"context": doc.page_content, "input": question}
            for doc in cleaned_documents
        ]

        # 준비된 입력을 사용하여 일괄 검사를 수행합니다.
        checked_results = groundedness_check.batch(checking_inputs)

        # 검사 결과가 'yes'인 문서만 필터링합니다.
        filtered_documents = [
            doc
            for doc, result in zip(cleaned_documents, checked_results)
            if result.score == "yes"
        ]
    else:
        # 검사기를 사용하지 않는 경우, 모든 정제된 문서를 그대로 사용합니다.
        filtered_documents = cleaned_documents

    # 필터링된 문서를 반환합니다.
    return filtered_documents
retrieve_and_check("디지털 정부혁신의 주요 목표는 무엇인가요?")







# 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

# Direction:
Make sure you understand the intent of the question and provide the most appropriate answer.
- Ask yourself the context of the question and why the questioner asked it, think about the question, and provide an appropriate answer based on your understanding.
2. Select the most relevant content (the key content that directly relates to the question) from the context in which it was retrieved to write your answer.
3. Create a concise and logical answer. When creating your answer, don't just list your selections, but rearrange them to fit the context so they flow naturally into paragraphs.
4. If you haven't searched for context for the question, or if you've searched for a document but its content isn't relevant to the question, you should say ‘I can't find an answer to that question in the materials I have’.
5. Write your answer in a table of key points.
6. Your answer must include all sources and page numbers.
7. Your answer must be written in Korean.
8. Be as detailed as possible in your answer.
9. Begin your answer with ‘This answer is based on content found in the document **📚’ and end with ‘**📌 source**’.
10. Page numbers should be whole numbers.

#Context: 
{context}

###

#Example Format:
**📚 문서에서 검색한 내용기반 답변입니다**

(brief summary of the answer)
(include table if there is a table in the context related to the question)
(include image explanation if there is a image in the context related to the question)
(detailed answer to the question)

**📌 출처**
[here you only write filename(.pdf only), page]

- 파일명.pdf, 192쪽
- 파일명.pdf, 192쪽
- ...

###

#Question:
{question}

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 단계 8: 체인(Chain) 생성
chain = (
    {"context": RunnableLambda(retrieve_and_check), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


print(chain.invoke("디지털 정부혁신의 주요 목표는 무엇인가요?"))


print(chain.invoke("디지털 취약계층을 위한 지원은 어떤 것들이 있나요?"))


print(chain.invoke("생애 주기 패키지 분야에 대해 설명하세요"))