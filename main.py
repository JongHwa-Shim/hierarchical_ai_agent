from dotenv import load_dotenv
load_dotenv()

from hie_agent.nodes import MainAgentNode, SubAgentNode
from hie_agent.tools.default_tools import *
from hie_agent.tools.custom_tools import *
from hie_agent.subagents.web_browsing.subagent import BrowserAgent
from hie_agent.utils import print_hie_agent_structure


import asyncio
from contextlib import AsyncExitStack

async def main():
        # Load environment variables from .env file
        
        # Initialize the async exit stack
        stack = AsyncExitStack()

        # load native tools
        document_loader = DocumentLoaderTool()
        img_generator = ImageGenerationTool(api_key=os.getenv("ImageGenerationTool_API_KEY"))
        img_qa = ImageQATool(api_key="ImageQATool_API_KEY")
        google_search = GoogleSearchTool()
        duckduckgo_search = DuckDuckGoSearchTool()
        visit_webpage = VisitWebpageTool()
        decryption_tool = DecryptionTool()
        hwp_reader = ReadHwpTool()
        pdf_summarizer = PDFFirstPageSummarizerTool()
        pdf_parser = PDFParserWithPreservedLayoutTool()

        # content = pdf_parser("./.data/documents/연구계획서.pdf")
        # print(f"[Parsed Results]\n{content}")


        # load subagents
        search_agent = await SubAgentNode.create(name="search_agent", 
                            description="""A managed agent that will search the internet to answer your question.
Ask him for all your questions that require Complex and detailed browsing the web.
Provide him as much context as possible, in particular if you need to search on a specific timeframe!
And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.""", 
                            tools=[],
                            sub_agents=[],
                            mcp_servers=["brave-search"], 
                            async_exit_stack=stack)

        browser_agent = await BrowserAgent.create(name=BrowserAgent.name, 
                            description=BrowserAgent.description, 
                            tools=[], 
                            sub_agents=[], 
                            mcp_servers=[], 
                            async_exit_stack=stack)
        
        DB_agent = await SubAgentNode.create(name="DB_agent", 
                            description="""A powerful and versatile agent for all your local files and knowledge management needs. This agent can read, store, query, and retrieve information from your permitted file systems and a specialized document database (Chroma DB). This Agent doesnt give up and do its best to explore the documents or DB.

Call this agent whenever you need to:
- Access content from files on a disk.
- Save new information or documents into a persistent knowledge base.
- Search for specific details, facts, or entire documents within your accumulated knowledge.
- Manage your document collection, ensuring information is readily available for retrieval.

Provide a clear and precise request describing what information you need, what document to process, or what query to perform. Focus on the -"what" and let the agent handle the "how." For instance, instead of detailing file paths or database operations, simply ask "Find information about X in the project documents" or "Save this report." However, if you want to navigate to a specific folder or file, be sure to provide the detailed file path! Do not omit any information related to the file path, provide all of it!""", 
                            tools=[hwp_reader, pdf_parser],
                            sub_agents=[],
                            mcp_servers=["chroma", "filesystem"],
                            async_exit_stack=stack)
        
        PDF_analysis_agent = await SubAgentNode.create(name="PDF_parser_agent", 
                            description="""This agent is a specialized agent that parses pdf files and extracts their internal contents. It also has the basic ability to navigate directories. Ask this agent for any tasks related to extracting pdf file contents! If you need to look through or parse the contents of multiple PDF documents, use a for loop to avoid repeating too many Thought-Code steps.""", 
                            tools=[pdf_parser, pdf_summarizer],
                            sub_agents=[],
                            mcp_servers=["filesystem", "pdf-parser"],
                            async_exit_stack=stack)
        # load main agent
        main_agent = await MainAgentNode.create(tools=[img_generator, img_qa, google_search], 
                                                sub_agents=[browser_agent, PDF_analysis_agent], 
                                                mcp_servers=[], 
                                                async_exit_stack=stack)
        
        # print the agent's hierarchy structure
        print_hie_agent_structure(main_agent)

        # Request for work
        final_result = await main_agent("2025년 2분기~4분기의 트럼프 2기 행정부의 관세 정책에 따른 삼성 전자 HBM 사업부의 영향 및 향후 전망에 대해 설명해줘. 다양한 웹사이트에서 정보를 살펴보고, 깊게 살펴봐야하는 홈페이지를 최소 5개 이상 크롤링을 통해서 심층적으로 조사하고, 너가 삼성 전자 HBM 사업부의 전문가이며 상사에게 보고하기 위해 글을 작성한다고 가정하고 향후 사업부 정책을 결정하기 위한 심층 조사 보고서의 형태로 작성해줘. 너의 능력을 시험해 보고 싶으니 firecrawl_deep_research 도구는 쓰지말아줘. 보고서도 최소 A4 4장 이상 분량으로 상세하게 작성해줘. 보고서에는 작성에 참고한 웹사이트 정보(링크 및 웹사이트 이름, 날짜)도 주석달아줘. 보고서는 기본 저장소에 저장해줘. 모든 작업을 완료했다고 가정하지 마시오. 모든 작업 수행과정은 한국어로 출력해.")

        # final_result = await main_agent("주어진 5개의 숫자 1, 5, 7, 3, 9 에서 4개씩 더할 때 가장 큰 숫자와 가장 작은 숫자를 찾아줘.")

        # final_result = await main_agent("내 로컬 폴더에서 style gan 관련 논문파일들을 찾고 해당 파일들을 DB에 저장해줘.")
        # final_result = await main_agent("내 DB에서 style gan 관련 내용을 검색하고 그 내용을 기반으로 style gan 논문의 저자가 누구인지 알려줘.")
        # final_result = await main_agent("2025년 4월 한달간의 평균 krw-usd 매매기준율을 찾아서 계산해줘. 웹사이트 검색을 통해서 실제 정보를 기반으로 답변해줘. maximum context length(200,000토큰)가 초과하지 않게 효율적으로 검색하고, 출력을 쓸데없이 많이 출력하지 마.")
        
        # final_result = await main_agent("내 research 폴더에서 국외여비 관련 정보가 들어있는 pdf파일들을 찾고 해당 파일의 내용을 분석해서 부교수가 미국으로 2박3일 출장을 간다면 여비를 얼마 청구할 수 있는지 계산해서 알려줘.")
        # print the work results
        print(final_result)

        await stack.aclose()

if __name__ == "__main__":
    # pdf_layout_parser = PDFParserWithPreservedLayoutTool()
    # a = "./.data/documents/research/example_pdf1.pdf"
    # x = pdf_layout_parser(a)
    # print(x)
    # a= 1      

    asyncio.run(main())
    # TODO: subagent의 description과 instruction을 각각 분리해도 좋을지도. description은 상위 에이전트에게 subagent의 역할을 설명하는 용도이고, instruction은 subagent의 task prompt에 입력되어 subagent의 행동을 결정하는 용도.
    # TODO: memory 요약 기능
