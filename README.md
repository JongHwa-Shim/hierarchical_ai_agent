# 이슈
1. 아래의 환경변수명은 원래 LLM_API_KEY로, 다른 코드에서도 계속 LLM_API_KEY라는 이름으로 사용되는데, 여기서만 변수명이 변경되어 있고 다른 부분은 반영되어 있지 않습니다. 환경변수라서 이름에 따라 에러가 발생하거나 그럴거 같지는 않은데 바꾸신 이유가 있을까요?
```
echo "OPEN_API_KEY=[openai_api_key]" > .env
```

2. pdfminer.six 패키지 버전을 20240706에서 20221105로 바꾸셨습니다. 
바꾸신 이유가 "HOCRConverter"라는 class를 찾지 못하는 오류 때문이라고 말씀하셨는데, 저는 오히려 20240706버전에서 HOCRConverter class가 있어서 오류가 안나고, 20221105 버전에서 오류가 납니다... 그래서 현재 버전은 20240706으로 사용하고 있습니다.

# 환경 설치 및 세팅
## UV 환경 설치
### UV 설치
```
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```
# On Mac
brew install uv 
```
### UV 패키지 환경 세팅
```
uv python list # 지원 가능한 파이썬 버전 확인
uv python install 3.11.12 # 설치가 안되어 있으면 설치
uv sync # uv 환경 동기화, .venv 폴더 생성됨
(실행) uv run main.py

(pyproject.toml이 없는 경우)
uv init --python=3.11.12 # uv 환경 초기화, pyproject.toml 생성됨
uv add -r requirements.txt # requirements.txt 패키지 설치, .venv 폴더 생성됨

```

## 아나콘다 환경 설치
```
conda create -n [env_name]
pip install -r requirements.txt
```
## 아나콘다 환경 작동 및 경로 설정
```
cd [project_root]
conda activate [env_name]
```

# 환경 변수 설정
- .env파일을 최상위 디렉토리에 만들어야 합니다.(.env.example 파일 참고)
- 아래는 무시
```
echo "LLM_API_KEY=[openai_api_key]" > .env # LLM_API_KEY이름에 종속되는 코드들이 많습니다. 이름은 LLM_API_KEY로 고정해야될 것 같습니다.
(아래 key는 https://www.searchapi.io/ 가입 후 api key 발급받기)
echo "SERPAPI_API_KEY=[serpapi_api_key]" >> .env
```

# 사용 예시 
```
# main.py 참조

from hie_agent.nodes import MainAgentNode, SubAgentNode
from hie_agent.tools.custom_tools import CustomTool1, CustomTool2, CustomTool3, CustomTool4

from dotenv import load_dotenv

load_dotenv() # ./.env 파일에서 환경 변수 설정 읽어옴.

# 사용 툴 인스턴스 정의
custom_tool_1 = CustomTool1()
custom_tool_2 = CustomTool2()

custom_tool_3 = CustomTool3()
custom_tool_4 = CustomTool4()

# 미리 사용자 정의된 서브 에이전트를 사용하지 않고, 즉석으로 서브 에이전트 만듬. 
# CustomTool3와 CustomTool4를 하위 툴로 사용하는 서브 에이전트.
sub_agent_1 = SubAgentNode(tools=[custom_tool_3, custom_tool_4])

# 메인 에이전트 인스턴스 선언
# 사용할 툴과 서브 에이전트 입력
# main_agent는 custom_tool_1, custom_tool_2를 하위 툴로, sub_agent_1을 하위 에이전트로 가짐.
main_agent = MainAgentNode(tools=[custom_tool_1, custom_tool_2], sub_agents=[sub_agent_1])

# 메인 에이전트에 요청 업무 입력
final_result = main_agent("IT 기업 로이드케이에 대해 조사해주고, 로이드케이의 이미지에 걸맞은 로고 이미지를 생성해줘.")

# 최종 결과 출력
print(final_result)
```
## 사용 예시 2 (사용자 정의된 서브 에이전트 사용)
```
# main.py 참조

from hie_agent.nodes import MainAgentNode
from hie_agent.tools.custom_tools import CustomTool1, CustomTool2
from hie_agent.subagents.custom_subagents import CustomSubAgnet1 # 사용자 정의된 서브 에이전트

from dotenv import load_dotenv

load_dotenv()

# 사용 툴 인스턴스 정의
custom_tool_1 = CustomTool1()
custom_tool_2 = CustomTool2()

# 사용 서브 에이전트 인스턴스 정의
# CustomSubAgent1은 미리 사용자 정의된 서브 에이전트로서 하위 툴이나 하위 에이전트가 미리 정의되어 있음. 
# (사용자 정의 SubAgent 만들기 예시 부분 참조)
custom_sub_agent_1 = CustomSubAgent1()

# 메인 에이전트 인스턴스 선언
# 사용할 툴과 서브 에이전트 입력
# main_agent는 custom_tool_1, custom_tool_2를 하위 툴로, custom_sub_agent_1을 하위 에이전트로 가짐.
main_agent = MainAgentNode(tools=[custom_tool_1, custom_tool_2], sub_agents=[custom_sub_agent_1])

# 메인 에이전트에 요청 업무 입력
final_result = main_agent("IT 기업 로이드케이에 대해 조사해주고, 로이드케이의 이미지에 걸맞은 로고 이미지를 생성해줘.")

# 최종 결과 출력
print(final_result)
```
# Tool 및 Subagent 만들기
## 사용자 정의 Tool 만들기 예시 
```
# .\hie_agent\tools\custom_tools.py 참조
# 새로운 사용자 정의 tool은 .\hie_agent\tools\custom_tools.py에 만드는 것을 추천

class Tool1(ToolNode):
    name = "name of this tool."
    description= "this is description of this tool"
    input_format = {'var1': {'type': 'string', 'description': 'this is description of var1.'}, 'var2': {'type': 'int', 'description': 'this is description of var2'}}
    output_format = [{'type': 'string', 'description': 'this is description of 1st output'}, {'type': 'bool', 'description': 'this is description of 2nd output'}]

    def __call__(self, var1, var2): # input_format에 선언된 입력변수와 이름과 갯수 동일하게 맞춰야 함
        # 원하는 기능 삽입
        return # output_format에 선언된 출력변수와 갯수 동일하게 맞춰야 함
```

## 사용자 정의 SubAgent 만들기 예시 
```
# 이미 만들어진 subagent는.\hie_agent\subagents\web_browsing\subagent.py 참조
# 새로운 사용자 정의 subagent는 .\hie_agent\subagents\custom_subagent.py 에 만드는 것을 추천

from ..tools.custom_tools import Tool1, Tool2, Tool3
from . import CustomSubAgent2, CustomSubAgent3

class CustomSubAgent1(SubAgentNode):
    name = "name of this sub agent."
    description = "this is description of this sub agent."

    # 사용자 정의 config 사용 (Optional, 아래 에이전트 설정 참조)
    config = SUB_AGENT_CUSTOM_CONFIG

    # subagent에 사용될 툴 인스턴스 입력
    basic_tools = [
        Tool1()
        Tool2()
        Tool3()
    ]

    # subagent에 사용될 서브에이전트 인스턴스 입력
    basic_sub_agents = [
        CustomSubAgent2()
        CustomSubAgnet3()
    ]
```
# 에이전트 설정
## 에이전트 설정 파일
```
# .\hie_agent\config.py

# MainAgent 설정
MAIN_AGENT_DEAFULT_CONFIG = {"print_process": True, "streaming": False, "planning_feedback": True, "react_step_limit": False, "max_react_step": 20, 
                             "llm_lib": "langchain", "llm_name": "o1", "max_tokens": 10000, "reasoning_effort": "high",
                             "llm_api_key": os.getenv("LLM_API_KEY")}

# SubAgent 설정 (모든 서브에이전트에 일괄적으로 적용)
SUB_AGENT_DEAFULT_CONFIG = {"print_process": True, "streaming": False, "planning_feedback": False, "react_step_limit": False, "max_react_step": 20, 
                             "llm_lib": "langchain", "llm_name": "o1", "reasoning_effort": "high", "max_tokens": 10000, 
                             "llm_api_key": os.getenv("LLM_API_KEY")}

# TODO print_process, straming 기능 구현 안됨
```
1. planning_feedback: True면 에이전트가 계획 수립후 사용자 피드백을 입력받음. 반복하여 계획 수정할 수 있음.
2. react_step_limit: 정수 숫자. 에이전트가 설정된 횟수 안으로 ReAct step을 끝내려고 노력함. 절대적 보장 x. False로 설정하면 기능 off 됨.
3. max_react_step: 정수 숫자. 에이전트의 ReAct step이 이 숫자 이상으로 초과되면 에이전트는 강제 중지됨. 그리고 상위 에이전트에게 ReAct step이 초과됐다는 메세지와 함께 에이전트의 작동 이력을 함께 반환함.
4. llm_lib, llm_name, max_tokens, reasoning_effort, temperature, llm_api_key: 현재 에이전트는 langchain을 사용하여 openai llm들을 사용함. 관련 설정 변수들.

- 새로운 설정 만들어서 특정 서브 에이전트에만 적용시킬 수도 있음. (사용자 정의 subagent 만들기 예시 참조)
