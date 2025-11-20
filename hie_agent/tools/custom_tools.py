# from hie_agent.nodes import ToolNode
from ..nodes import ToolNode
import os
from typing import Optional
import requests

class ImageGenerationTool(ToolNode):
    name = "image_generator"
    description = "This tool takes an input text description and produces an image matching that description."
    input_format = {'prompt': {'type': 'string', 'description': 'Text description used to generate the image'}}
    output_format = [{'type': 'PIL.Image', 'description': 'PIL.Image instance of the generated image.'}]

    def __init__(self, api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        
    def __call__(self, prompt: str):
        from PIL import Image
        import io
        import requests
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        img_bytes = requests.get(response.data[0].url).content
        img_pil = Image.open(io.BytesIO(img_bytes))

        # img_pil.save("./img.png")

        return img_pil

class ImageQATool(ToolNode):
    name = "image_recognizer"
    description = "This tool takes an image and a question about the image and outputs an answer. It allows for visual analysis and recognition of images."
    input_format = {'image_path': {'type': 'string', 'description': 'Path to the image.'}, 'question': {'type': 'string', 'description': 'Questions about images.'}}
    output_format = [{'type': 'string', 'description': 'Answers to questions about images.'}]

    def __init__(self, api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
    
    def __call__(self, image_path: str, question: Optional[str]= None):
        import base64

        if question is None:
            question = "Please write a detailed caption for this image."
        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        # Getting the Base64 string
        base64_image = encode_image(image_path)

        response = self.client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": f"{question}" },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
        
        return response.output_text
    
class DocumentLoaderTool(ToolNode):
    name = "document_loader"
    description = """This tool loads and reads various types of document files and returns the text content. Document files with the extensions below should be loaded using this tool.
This tool handles the following file extensions: [".docx"]"""
    input_format = {'file_path': {'type': 'string', 'description': "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'."}}
    output_format = [{'type': 'string', 'description': 'Text content of the document file.'}]

    def __init__(self):
        pass

    def __call__(self, file_path):
        file_ext = file_path.split(".")[-1]

        if file_ext == "docx":
            from langchain_community.document_loaders import Docx2txtLoader

            loader = Docx2txtLoader(file_path)

            docs = loader.load()

            return docs[0].page_content
        
class DuckDuckGoSearchTool(ToolNode):
    name = "duckduckgo_search"
    description = """Performs a duckduckgo web search based on your query (think a Google search) then returns the short string snippets of top search results. If you need some light information search, call this tool"""
    input_format = {"query": {"type": "string", "description": "The search query to perform."}}
    output_format = [{'type': 'string', 'description': 'Returns 10 short DuckDuckGO search result snippets for a query.'}]

    def __init__(self, max_results=10, **kwargs):
        self.max_results = max_results
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
            ) from e
        self.ddgs = DDGS(**kwargs)
    
    def __call__(self, query):
        results = self.ddgs.text(query, max_results=self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)

class GoogleSearchTool(ToolNode):
    name = "google_search"
    description = """Performs a google web search for your query then returns a short string snippets of the top search results. If you need some light information search, call this tool."""
    input_format = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_format = [{'type': 'string', 'description': 'Returns 1 to 10 short Google search result snippets for a query.'}]

    def __init__(self, provider: str = "serpapi"):
        super().__init__()
        import os

        self.provider = provider
        if provider == "serpapi":
            self.organic_key = "organic_results"
            api_key_env_name = "SERPAPI_API_KEY"
        else:
            self.organic_key = "organic"
            api_key_env_name = "SERPER_API_KEY"
        self.api_key = os.getenv(api_key_env_name)
        if self.api_key is None:
            raise ValueError(f"Missing API key. Make sure you have '{api_key_env_name}' in your env variables.")

    def __call__(self, query: str, filter_year: Optional[int] = None) -> str:
        import requests

        if self.provider == "serpapi":
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "google_domain": "google.com",
            }
            base_url = "https://serpapi.com/search.json"
        else:
            params = {
                "q": query,
                "api_key": self.api_key,
            }
            base_url = "https://google.serper.dev/search"
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            results = response.json()
        else:
            raise ValueError(response.json())

        if self.organic_key not in results.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")
        if len(results[self.organic_key]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets = []
        if self.organic_key in results:
            for idx, page in enumerate(results[self.organic_key]):
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                web_snippets.append(redacted_version)

        return "## Search Results\n" + "\n\n".join(web_snippets)

class VisitWebpageTool(ToolNode):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. If you need to do a simple web page search, use this to navigate the web."
    )
    input_format = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_format = [{'type': 'string', 'description': ''}]

    def __init__(self, max_output_length: int = 40000):
        super().__init__()
        self.max_output_length = max_output_length

    def __call__(self, url: str) -> str:
        try:
            import re

            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException

            from smolagents.utils import truncate_content
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e
        try:
            # Send a GET request to the URL with a 20-second timeout
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return truncate_content(markdown_content, self.max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
        
class DecryptionTool(ToolNode):
    name = "decryption_tool"
    description = (
        "This tool takes crypto, decrypts it, and return decryption results."
    )
    input_format = {
        "crypto": {
            "type": "string",
            "description": "encrypted string",
        }
    }
    output_format = [{'type': 'string', 'description': 'decryption results'}]

    def __call__(self, crypto):
        if crypto == "asdfzxcv":
            return "안녕하세요! 암호해독을 성공하셨습니다! 축하드립니다!"
        
class ReadHwpTool(ToolNode):
    name = "hwp_reader"
    description = (
        "Reads all text content of a file with the .hwp or .hwpx extension (Hancom file) and returns it as a string. Use this tool if you need to open a Hancom file!"
    )
    input_format = {
        "hwp_file_path": {
            "type": "string",
            "description": "path of the hwp file",
        }
    }
    output_format = [{'type': 'string', 'description': 'parsed text of hwp file'}]

    def __init__(self):
        super().__init__()

        import olefile
        import zlib
        import struct
        #### 추가 ####
        import re
        import unicodedata

        class HWPExtractor(object):
            FILE_HEADER_SECTION = "FileHeader"
            HWP_SUMMARY_SECTION = "\x05HwpSummaryInformation"
            SECTION_NAME_LENGTH = len("Section")
            BODYTEXT_SECTION = "BodyText"
            HWP_TEXT_TAGS = [67]

            def __init__(self, filename):
                self._ole = self.load(filename)
                self._dirs = self._ole.listdir()

                self._valid = self.is_valid(self._dirs)
                if (self._valid == False):
                    raise Exception("Not Valid HwpFile")
                
                self._compressed = self.is_compressed(self._ole)
                self.text = self._get_text()
            
            # 파일 불러오기 
            def load(self, filename):
                return olefile.OleFileIO(filename)
            
            # hwp 파일인지 확인 header가 없으면 hwp가 아닌 것으로 판단하여 진행 안함
            def is_valid(self, dirs):
                if [self.FILE_HEADER_SECTION] not in dirs:
                    return False

                return [self.HWP_SUMMARY_SECTION] in dirs

            # 문서 포맷 압축 여부를 확인
            def is_compressed(self, ole):
                header = self._ole.openstream("FileHeader")
                header_data = header.read()
                return (header_data[36] & 1) == 1

            # bodytext의 section들 목록을 저장
            def get_body_sections(self, dirs):
                m = []
                for d in dirs:
                    if d[0] == self.BODYTEXT_SECTION:
                        m.append(int(d[1][self.SECTION_NAME_LENGTH:]))

                return ["BodyText/Section"+str(x) for x in sorted(m)]
            
            # text를 뽑아내는 함수
            def get_text(self):
                return self.text

            # 전체 text 추출
            def _get_text(self):
                sections = self.get_body_sections(self._dirs)
                text = ""
                for section in sections:
                    text += self.get_text_from_section(section)
                    text += "\n"

                self.text = text
                return self.text

            # section 내 text 추출
            def get_text_from_section(self, section):
                bodytext = self._ole.openstream(section)
                data = bodytext.read()

                unpacked_data = zlib.decompress(data, -15) if self.is_compressed else data
                size = len(unpacked_data)

                i = 0

                text = ""
                while i < size:
                    header = struct.unpack_from("<I", unpacked_data, i)[0]
                    rec_type = header & 0x3ff
                    level = (header >> 10) & 0x3ff
                    rec_len = (header >> 20) & 0xfff

                    if rec_type in self.HWP_TEXT_TAGS:
                        rec_data = unpacked_data[i+4:i+4+rec_len]
                        
                        ############## 정제 추가된 부분 #############
                        decode_text = rec_data.decode('utf-16')
                        # 문자열을 담기 전 정제하기
                        res = self.remove_control_characters(self.remove_chinese_characters(decode_text))
                        
                        text += res
                        text += "\n"

                    i += 4 + rec_len

                return text
        #################### 텍스트 정제 함수 #######################
            # 중국어 제거
            def remove_chinese_characters(self, s: str):   
                return re.sub(r'[\u4e00-\u9fff]+', '', s)
                
            # 바이트 문자열 제거
            def remove_control_characters(self, s):    
                return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")
            
        self.hwp_extractor = HWPExtractor

    def __call__(self, hwp_file_path):
        hwp = self.hwp_extractor(hwp_file_path)
        text = hwp.get_text()
        return text

class PDFFirstPageSummarizerTool(ToolNode):
    name = "pdf_summarizer"
    description = (
        "This tool parse pdf file and return the first page's text content. Use this tool if you want to get a quick overview of the contents of a pdf file!"
    )
    input_format = {
        "pdf_path": {
            "type": "string",
            "description": "path of the pdf file.",
        },
        "max_chars": {
            "type": "integer",
            "description": "maximum number of characters to return from the first page's text content. Default is 200.",
        }
    }
    output_format = [{'type': 'string', 'description': 'parsed text of pdf file'}]

    def get_first_page_summary(self, pdf_path: str, max_chars: int = 200) -> str:
        """
        PDF 파일의 첫 페이지 텍스트 내용의 앞부분을 지정된 문자 수만큼 반환합니다.

        Args:
            pdf_path (str): PDF 파일 경로.
            max_chars (int, optional): 반환할 최대 문자 수. 기본값은 200입니다.

        Returns:
            str: 첫 페이지 텍스트의 요약 또는 오류 메시지.
        """
        import fitz  # PyMuPDF
        import os

        if not os.path.exists(pdf_path):
            return f"오류: 파일 '{pdf_path}'를 찾을 수 없습니다."
        if not os.path.isfile(pdf_path):
            return f"오류: '{pdf_path}'는 파일이 아닙니다."

        try:
            # PDF 파일 열기
            doc = fitz.open(pdf_path)

            if len(doc) == 0:
                doc.close()
                return "정보: PDF에 페이지가 없습니다."

            # 첫 번째 페이지 가져오기
            first_page = doc[0]

            # 페이지에서 텍스트 추출
            text = first_page.get_text("text") # "text", "html", "xml", "xhtml" 등 가능

            # 문서 닫기
            doc.close()

            if not text.strip(): # .strip()으로 공백만 있는 경우도 처리
                return "정보: 첫 페이지에 추출할 수 있는 텍스트가 없습니다."

            # 불필요한 공백(줄바꿈, 연속 공백 등) 정리
            cleaned_text = " ".join(text.split())

            # 지정된 문자 수만큼 자르기
            if len(cleaned_text) > max_chars:
                summary = cleaned_text[:max_chars] + "..."
            else:
                summary = cleaned_text
            
            return summary

        except fitz.fitz.PasswordError: # PyMuPDF 1.18.11 이후로는 fitz.PasswordError
            return "오류: PDF가 암호로 보호되어 있어 내용을 읽을 수 없습니다."
        except RuntimeError as e: # PyMuPDF 관련 런타임 에러 (예: 손상된 파일)
            return f"오류: PDF 처리 중 문제가 발생했습니다 - {e}"
        except Exception as e:
            return f"예상치 못한 오류가 발생했습니다: {e}"

    def __call__(self, pdf_path, max_chars):
        return self.get_first_page_summary(pdf_path, max_chars=max_chars)
    
import pdfplumber
import io
import traceback
import re # 정규 표현식 모듈 임포트
class PDFParserWithPreservedLayoutTool(ToolNode):
    name = "pdf_parser"
    description = (
        "This tool parses the entire text content of PDF file while preserving the PDF layout, including tables. Use this tool when you want to strictly extract the whole contents of a file. This tool provide table content as markdown format."
    )
    input_format = {
        "pdf_path": {
            "type": "string",
            "description": "path of the pdf file.",
        }
    }
    output_format = [{'type': 'string', 'description': 'Parsed text of pdf file with layout(ex. tables) preserved'}]        
    def __init__(self):
        super().__init__()

    def preprocess_table_for_markdown(self, table_data):
        if not table_data:
            return []
        num_rows = len(table_data)
        if num_rows == 0:
            return []
        num_cols = 0
        for r_idx, row_content in enumerate(table_data):
            if row_content is not None:
                current_cols = len(row_content)
                if current_cols > num_cols:
                    num_cols = current_cols
        if num_cols == 0:
            return [["" for _ in range(len(row) if row else 0)] for row in table_data]
        processed_table = []
        for r_idx in range(num_rows):
            row_content = table_data[r_idx]
            if row_content is None:
                processed_table.append([None] * num_cols)
            else:
                new_row = list(row_content)
                while len(new_row) < num_cols:
                    new_row.append(None)
                processed_table.append(new_row[:num_cols])
        for c_idx in range(num_cols):
            for r_idx in range(num_rows):
                if processed_table[r_idx][c_idx] is None:
                    value_to_propagate = None
                    for prev_r_idx in range(r_idx - 1, -1, -1):
                        if processed_table[prev_r_idx][c_idx] is not None:
                            value_to_propagate = processed_table[prev_r_idx][c_idx]
                            break
                    if value_to_propagate is not None:
                        processed_table[r_idx][c_idx] = value_to_propagate
        for r_idx in range(num_rows):
            for c_idx in range(num_cols):
                if processed_table[r_idx][c_idx] is None:
                    value_to_propagate = None
                    for prev_c_idx in range(c_idx - 1, -1, -1):
                        if processed_table[r_idx][prev_c_idx] is not None:
                            value_to_propagate = processed_table[r_idx][prev_c_idx]
                            break
                    if value_to_propagate is not None:
                        processed_table[r_idx][c_idx] = value_to_propagate
        final_table = [[str(cell).strip() if cell is not None else "" for cell in row] for row in processed_table]
        return final_table

    def table_to_markdown_format(self, table_data):
        if not table_data:
            return ""
        num_cols = len(table_data[0]) if table_data and table_data[0] else 0
        if num_cols == 0:
            if any(table_data):
                num_cols = max(len(row) for row in table_data if row) if any(len(row) for row in table_data if row) else 0
            if num_cols == 0: return ""
        uniform_table = []
        for row_data in table_data:
            new_row = list(row_data)
            while len(new_row) < num_cols:
                new_row.append("")
            uniform_table.append(new_row[:num_cols])
        if not uniform_table: return ""
        header = "|" + "|".join(str(cell).replace("\n", " ").replace("|", "\\|") for cell in uniform_table[0]) + "|"
        separator = "|" + "|".join(["---"] * num_cols) + "|"
        body_lines = []
        for row in uniform_table[1:]:
            body_lines.append("|" + "|".join(str(cell).replace("\n", " ").replace("|", "\\|") for cell in row) + "|")
        return "\n".join([header, separator] + body_lines)

    def is_within_bboxes(self, obj_bbox, bboxes): # 지금 알고리즘에서는 사용안됨
        obj_x0, obj_top, obj_x1, obj_bottom = obj_bbox
        for bbox in bboxes:
            tbl_x0, tbl_top, tbl_x1, tbl_bottom = bbox
            if not (obj_x1 < tbl_x0 or obj_x0 > tbl_x1 or obj_bottom < tbl_top or obj_top > tbl_bottom):
                return True
        return False

    def clean_text_block(self, text_block):
        """추출된 텍스트 블록에서 여백을 정리, 중요한 줄 바꿈은 유지"""
        if not text_block:
            return ""
        
        lines = text_block.splitlines()
        cleaned_lines = []
        for line in lines:
            # 각 줄의 앞뒤 공백 제거, 여러 공백을 하나로
            normalized_line = ' '.join(line.split())
            if normalized_line: # 내용이 있는 줄만 추가
                cleaned_lines.append(normalized_line)
                
        # 문단 구분을 위해, 원래 여러 줄 바꿈이 있던 곳을 두 번의 줄 바꿈으로 표현 시도 (조정 필요)
        # 현재는 각 비어있지 않은 줄을 한 줄로 처리하고 있음
        return "\n".join(cleaned_lines)


    def extract_content_from_pdf(self, pdf_path):
        all_content_parts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    all_content_parts.append(f"\n## Page {page_num + 1}\n")
                    page_elements = []

                    table_settings = {
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "text_x_tolerance": 3,
                        "text_y_tolerance": 3,
                        "intersection_x_tolerance": 3,
                        "intersection_y_tolerance": 3,
                    }
                    found_tables_objects = page.find_tables(table_settings=table_settings)
                    sorted_tables = sorted(found_tables_objects, key=lambda t: t.bbox[1])

                    for tbl_obj in sorted_tables:
                        if tbl_obj.bbox:
                            raw_table_data = tbl_obj.extract()
                            if not raw_table_data: continue
                            processed_data = self.preprocess_table_for_markdown(raw_table_data)
                            markdown_table = self.table_to_markdown_format(processed_data)
                            if markdown_table:
                                page_elements.append({
                                    "y0": tbl_obj.bbox[1],
                                    "x0": tbl_obj.bbox[0],
                                    "type": "table",
                                    "content": markdown_table,
                                    "bbox": tbl_obj.bbox
                                })
                    
                    page_width = page.width
                    page_height = page.height
                    
                    text_extraction_bboxes = []
                    table_bboxes_sorted = sorted([el["bbox"] for el in page_elements if el["type"] == "table"], key=lambda b: b[1])

                    last_y_bottom = 0
                    for tbl_bbox in table_bboxes_sorted:
                        if tbl_bbox[1] > last_y_bottom:
                            text_extraction_bboxes.append( (0, last_y_bottom, page_width, tbl_bbox[1]) )
                        last_y_bottom = tbl_bbox[3]
                    
                    if last_y_bottom < page_height:
                        text_extraction_bboxes.append( (0, last_y_bottom, page_width, page_height) )
                    
                    if not table_bboxes_sorted:
                        text_extraction_bboxes.append( (0, 0, page_width, page_height) )

                    for text_idx, bbox_to_crop in enumerate(text_extraction_bboxes):
                        if bbox_to_crop[2] <= bbox_to_crop[0] or bbox_to_crop[3] <= bbox_to_crop[1]:
                            continue

                        cropped_page = page.crop(bbox_to_crop)
                        # layout=False 로 설정하고, x_tolerance, y_tolerance 값 조정
                        text_from_crop = cropped_page.extract_text(x_tolerance=1, y_tolerance=1, layout=False) 
                        
                        cleaned_text_from_crop = self.clean_text_block(text_from_crop) # 새로 정의한 함수 사용

                        if cleaned_text_from_crop:
                            page_elements.append({
                                "y0": bbox_to_crop[1],
                                "x0": bbox_to_crop[0],
                                "type": "text",
                                "content": cleaned_text_from_crop,
                                "bbox": bbox_to_crop 
                            })
                    
                    page_elements.sort(key=lambda el: (el['y0'], el['x0']))
                    
                    current_content_string = ""
                    for elem_idx, element in enumerate(page_elements):
                        current_content_string += element['content']
                        # 요소들 사이에 적절한 줄 바꿈 추가 (특히 텍스트와 테이블 사이, 텍스트와 텍스트 사이)
                        if elem_idx < len(page_elements) - 1:
                            next_element = page_elements[elem_idx+1]
                            # 현재 요소나 다음 요소가 테이블이거나, y 좌표가 크게 다르면 두 번 줄 바꿈
                            if element['type'] == 'table' or next_element['type'] == 'table' or \
                            (next_element['y0'] - element.get('bbox', [0,0,0,element['y0']])[3] > 20): # y0차이가 크면 (bbox[3]은 현재 요소의 bottom)
                                current_content_string += "\n\n"
                            else: # 텍스트와 텍스트 사이는 한 번 줄 바꿈
                                current_content_string += "\n" 
                    
                    all_content_parts.append(current_content_string)
                    all_content_parts.append("\n---") # 페이지 구분선

        except Exception as e:
            error_message = f"Error processing PDF: {str(e)}\n"
            error_message += "Traceback:\n" + traceback.format_exc()
            return error_message
        
        # 최종적으로 합쳐진 문자열에서 연속된 빈 줄을 하나로 줄이고, 앞뒤 공백 제거
        final_text = "\n".join(all_content_parts)
        # 여러 빈 줄을 하나의 빈 줄로 (또는 아예 제거)
        final_text = re.sub(r'\n\s*\n', '\n\n', final_text) # 두 개 이상의 연속된 줄바꿈을 두 개로 표준화
        return final_text.strip()

    def __call__(self, pdf_path):
        return self.extract_content_from_pdf(pdf_path)