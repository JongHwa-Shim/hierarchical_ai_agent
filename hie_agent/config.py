import os

PROMPT_TEMPLATE_PATH = "hie_agent/prompts/prompts.yaml"

DEFAULT_LIBRARIES = ['asyncio', 'inspect', 'base64', 'mcp', 'Bio', 'PIL', 'PyPDF2', 'bs4', 'chess', 'collections', 'csv', 'datetime', 'fractions', 'io', 'itertools', 'json', 'math', 'numpy', 'os', 'pandas', 'pptx', 'pubchempy', 'pydub', 'queue', 'random', 're', 'requests', 'scipy', 'sklearn', 'stat', 'statistics', 'sympy', 'time', 'torch', 'unicodedata', 'xml', 'yahoo_finance', 'zipfile']

# DEFAULT_LIBRARIES = ['requests', 'bs4']

# Data storage related config
TEMP_STORAGE_PATH = r"./.data/storage/temp_storage"
FINAL_RESULT_STORAGE_PATH = r"./.data/storage/final_result_storage"

# mcp config
MCP_CONFIG_PATH = r"./mcp_server_config.json"

# agent config
MAIN_AGENT_DEAFULT_CONFIG = {"print_process": True, 
                             "streaming": False, 
                             "planning_feedback": True, 
                             "react_step_limit": False, 
                             "max_react_step": 20, 
                             "llm_lib": "langchain", 
                             "llm_name": "gpt-5-mini", # Allowed values: "gpt-4o", "o1", "gpt-5-mini" or "gpt-5-nano"
                             "max_tokens": 200000, 
                             "reasoning_effort": "medium", # Allowed values: "high", "medium", "low".
                             "temperature": 0.7, 
                             "llm_api_key": os.getenv("LLM_API_KEY")}

SUB_AGENT_DEAFULT_CONFIG = {"print_process": True, 
                            "streaming": False, 
                            "planning_feedback": False, 
                            "react_step_limit": False, 
                            "max_react_step": 100, 
                            "llm_lib": "langchain", 
                            "llm_name": "gpt-5-mini", # Allowed values: "gpt-4o", "o1", "gpt-5-mini" or "gpt-5-nano"
                            "reasoning_effort": "low", # Allowed values: "high", "medium", "low".
                            "temperature": 0.1, 
                            "max_tokens": 200000, 
                            "llm_api_key": os.getenv("LLM_API_KEY")}

# TODO print_process, straming, observation 축약 기능 구현 필요