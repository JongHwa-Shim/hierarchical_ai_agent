from hie_agent.nodes import ToolNode

class VisitTool(ToolNode):
    name = "visit_page"
    description = "Visit a webpage at a given URL and return its text. If you want to perform a browser-based search, first call this tool to open the web."
    input_format = {"url": {"type": "string", "description": "The relative or absolute url of the webpage to visit."}}
    output_format = [{'type': "string", "description": "text contents of visited webpage."}]

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def __call__(self, url: str) -> str:
        self.browser.visit_page(url)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content

class PageUpTool(ToolNode):
    name = "page_up"
    description = "Scroll the viewport UP one page-length in the current webpage and return the new viewport content."
    input_format = {}
    output_format = [{'type': 'string', 'description': 'the text contents of new viewport of webpage'}]

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def __call__(self) -> str:
        self.browser.page_up()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content
    
class PageDownTool(ToolNode):
    name = "page_down"
    description = (
        "Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content."
    )
    input_format = {}
    output_format = [{'type': 'string', 'description': 'the text contents of new viewport of webpage'}]

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def __call__(self) -> str:
        self.browser.page_down()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content

class FinderTool(ToolNode):
    name = "find_on_page_ctrl_f"
    description = "Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F."
    input_format = {
        "search_string": {
            "type": "string",
            "description": "The string to search for on the page. This search string supports wildcards like '*'",
        }
    }
    output_format = [{'type': 'string', 'description': 'The text contents of viewport of the webpage where the search phrase was first found.'}]

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, search_string: str) -> str:
        find_result = self.browser.find_on_page(search_string)
        header, content = self.browser._state()

        if find_result is None:
            return (
                header.strip()
                + f"\n=======================\nThe search string '{search_string}' was not found on this page."
            )
        else:
            return header.strip() + "\n=======================\n" + content

class FindNextTool(ToolNode):
    name = "find_next"
    description = "Scroll the viewport to next occurrence of the search string. This is equivalent to finding the next match in a Ctrl+F search."
    input_format = {}
    output_format = [{'type': 'string', 'description': 'The text contents of viewport of the webpage where the search phrase was found next.'}]

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        find_result = self.browser.find_next()
        header, content = self.browser._state()

        if find_result is None:
            return header.strip() + "\n=======================\nThe search string was not found on this page."
        else:
            return header.strip() + "\n=======================\n" + content

# from typing import Optional

# from smolagents import Tool
# from smolagents.models import MessageRole, Model

# from .mdconvert import MarkdownConverter

# class TextInspectorTool(ToolNode):
#     name = "inspect_file_as_text"
#     description = """
# You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
# This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

#     input_format = {
#         "file_path": {
#             "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
#             "type": "string",
#         },
#         "question": {
#             "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
#             "type": "string",
#             "nullable": True,
#         },
#     }
#     output_format = [{'type': 'string', 'description': ''}]
#     md_converter = MarkdownConverter()

#     def __init__(self, model: Model, text_limit: int):
#         super().__init__()
#         self.model = model
#         self.text_limit = text_limit

#     def forward_initial_exam_mode(self, file_path, question):
#         result = self.md_converter.convert(file_path)

#         if file_path[-4:] in [".png", ".jpg"]:
#             raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

#         if ".zip" in file_path:
#             return result.text_content

#         if not question:
#             return result.text_content

#         if len(result.text_content) < 4000:
#             return "Document content: " + result.text_content

#         messages = [
#             {
#                 "role": MessageRole.SYSTEM,
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Here is a file:\n### "
#                         + str(result.title)
#                         + "\n\n"
#                         + result.text_content[: self.text_limit],
#                     }
#                 ],
#             },
#             {
#                 "role": MessageRole.USER,
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Now please write a short, 5 sentence caption for this document, that could help someone asking this question: "
#                         + question
#                         + "\n\nDon't answer the question yourself! Just provide useful notes on the document",
#                     }
#                 ],
#             },
#         ]
#         return self.model(messages).content

#     def __call__(self, file_path, question: Optional[str] = None) -> str:
#         result = self.md_converter.convert(file_path)

#         if file_path[-4:] in [".png", ".jpg"]:
#             raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

#         if ".zip" in file_path:
#             return result.text_content

#         if not question:
#             return result.text_content

#         messages = [
#             {
#                 "role": MessageRole.SYSTEM,
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "You will have to write a short caption for this file, then answer this question:"
#                         + question,
#                     }
#                 ],
#             },
#             {
#                 "role": MessageRole.USER,
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Here is the complete file:\n### "
#                         + str(result.title)
#                         + "\n\n"
#                         + result.text_content[: self.text_limit],
#                     }
#                 ],
#             },
#             {
#                 "role": MessageRole.USER,
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
#                         + question,
#                     }
#                 ],
#             },
#         ]
#         return self.model(messages).content
