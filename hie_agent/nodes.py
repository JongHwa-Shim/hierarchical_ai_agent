from typing import TypedDict, List, Dict, Any, Optional, Union, Dict, Tuple, Callable, TypeVar, Generic
import yaml
import os
import textwrap
from jinja2 import Template, StrictUndefined
import sys
import asyncio
import json
from contextlib import AsyncExitStack
import mcp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from hie_agent.config import *
from hie_agent.code_executor import LocalPythonExecutor
from hie_agent.container import *
from hie_agent.utils import *

class Node:
    name: str

    def __init__(self):
        pass
    
class ToolNode(Node):
    name: str
    description: str = "this is description of this tool"
    input_format: Dict[str, Dict[str, Union[str, type, bool]]] # example: {'var1': {'type': 'string', 'description': 'this is description of var1.'}, 'var2': {'type': 'int', 'description': 'this is description of var2'}}
    output_format: List[Dict[str, Union[str, type, bool]]] # example: [{'type': 'string', 'description': 'this is description of 1st output'}, {'type': 'bool', 'description': 'this is description of 2nd output'}]

    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def _check_inputs(self):
        pass

    def _check_outputs(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

class MCPToolNode(Node):
    name: str
    description: str # example: "this is description of this tool"
    input_format: Dict[str, Dict[str, Union[str, type, bool]]] # example: {'var1': {'type': 'string', 'description': 'this is description of var1.'}, 'var2': {'type': 'int', 'description': 'this is description of var2'}}
    required_input: List[str] # example: ['var1', 'var2']
    output_format: List[Dict] = [{'description': "output of this MCP tool."}, {"description": "type of output"}] # example: [output, "<class 'mcp.types.TextContent'>"]
    mcp_client: ClientSession
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs["name"]
        self.description = kwargs["description"]
        self.input_format = kwargs["input_format"]
        self.required_input = kwargs["required_input"]
        self.mcp_client = kwargs["mcp_client"]
        
    def _check_inputs(self):
        pass

    def _check_outputs(self):
        pass

    async def __call__(self, *args, **kwargs):
        input_var_dict = self._handling_input(args, kwargs)
                
        mcp_response = await self.mcp_client.call_tool(self.name, input_var_dict)
        mcp_response = mcp_response.content[0] if mcp_response.content else None

        output, output_type = self._handling_output(mcp_response=mcp_response)

        return output, output_type

    def _handling_input(self, args, kwargs):
        # handling positional and keyword arguments
        input_var_dict = {}
        if args != []:
            args_len = len(args)
            input_format_dict = enumerate(self.input_format)
            for i in range(args_len):
                i, input_name = next(input_format_dict)
                input_var_dict[input_name] = args[i]
        if kwargs != {}:
            for key, value in kwargs.items():
                if key in self.input_format:
                    input_var_dict[key] = value
                else:
                    raise ValueError(f"Invalid input variable: {key}")
        
        return input_var_dict
    
    def _handling_output(self, mcp_response):
        # handling output type
        if isinstance(mcp_response, mcp.types.TextContent):
            output = mcp_response.text
            output_type = "mcp.types.TextContent"
        elif isinstance(mcp_response, mcp.types.ImageContent):
            output = [mcp_response.data, mcp_response.mimeType]
            output_type = "mcp.types.ImageContent"
        elif isinstance(mcp_response, mcp.types.EmbeddedResource):
            resource = mcp_response.resource
            if isinstance(resource, mcp.types.TextResourceContents):
                output = resource.text
                output_type = "mcp.types.TextResourceContents"
            elif isinstance(resource, mcp.types.BlobResourceContents):
                output = resource.blob
                output_type = "mcp.types.BlobResourceContents"
        else:
            output = mcp_response
            output_type = str(type(mcp_response))
        
        return output, output_type
    
class MainAgentNode(Node):
    name: str = "main_agent" # Cannot change
    config: Dict[str, Any] = MAIN_AGENT_DEAFULT_CONFIG
    basic_tools: List['ToolNode'] = []
    basic_sub_agents: List['SubAgentNode'] = []
    default_libraries = DEFAULT_LIBRARIES
    is_main_agent = True

    def __init__(self, config: Dict[str, Any]=None):
        super().__init__()
        # Renew config
        if config:
            self.config.update(config)
    
    @classmethod
    async def create(cls, tools: List['ToolNode']=[], sub_agents: List['SubAgentNode']=[], mcp_servers: List[str]=[], async_exit_stack: AsyncExitStack = None, allowed_libraries: List[str]=[], config: Dict[str, Any]=None):
        agent_ins = cls(config)

        await agent_ins._perform_async_init(tools=tools, sub_agents=sub_agents, mcp_servers=mcp_servers, async_exit_stack=async_exit_stack, allowed_libraries=allowed_libraries, config=config)

        return agent_ins

    async def _perform_async_init(self, tools: List['ToolNode']=[], sub_agents: List['SubAgentNode']=[], mcp_servers: List[str]=[], async_exit_stack: AsyncExitStack=None, allowed_libraries: List[str]=[], config: Dict[str, Any]=None):
        # init mcp tools, native tools, and sub_agents
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        await self._setup_mcp_tools(mcp_servers, async_exit_stack)
        self._setup_tools(tools)
        self._setup_sub_agents(sub_agents)

        # init allowed_libraries
        self._setup_allowed_libraries(allowed_libraries)

        # init python_executor
        # TODO: mcp 기능지원 추가해야함
        self.code_executor = LocalPythonExecutor(allowed_libraries=self.allowed_libraries, tools=self.tools, mcp_tools=self.mcp_tools, sub_agents=self.sub_agents) 

        # init temporary data stoarge
        self._setup_temp_storage()

        # init llm
        self.llm_engine = LLMContainer(**self.config)

        # init prompt templates
        self.prompt_templates = yaml.safe_load(open(PROMPT_TEMPLATE_PATH, 'r'))

        # make memory
        self.plan_memory = PlanMemory()
        self.react_memory = ReActMemory()
        self.react_step = 1

    async def _setup_mcp_tools(self, mcp_servers, async_exit_stack: AsyncExitStack=None):
        self.mcp_tools = {} # be used by cond executor
        self.mcp_server_info = {} # be used in constructing prompt

        for mcp_server in mcp_servers:
            self.mcp_server_info[mcp_server] = {"usage_guides": "", "tools": {}}
            # MCP 서버 설정 읽기
            with open(MCP_CONFIG_PATH) as f:
                config = json.load(f)["mcpServers"][mcp_server]
            server_params = StdioServerParameters(**config)
            
            # MCP 서버 이용 설명이 따로 있다면 가져오기 (이것은 MCP 서버에서 제공되지 않고 사용자가 구축해야 함.)
            try:
                self.mcp_server_info[mcp_server]["usage_guides"] = config["usage-guides"]
            except:
                pass

            # MCP 서버 실행 및 세션 초기화 
            stack = async_exit_stack
            stdio, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()

            mcp_tool_list = await session.list_tools()
            for mcp_tool in mcp_tool_list.tools:
                mcp_tool_node = MCPToolNode(name=mcp_tool.name, 
                                            description=mcp_tool.description, 
                                            input_format = mcp_tool.inputSchema.get('properties'), 
                                            required_input=mcp_tool.inputSchema.get('required'), 
                                            mcp_client=session) 
                                            # mcp tool들의 annotation 어트리뷰트는 입력되지 않음 (None인 경우가 대부분이라..)

                self.mcp_tools[mcp_tool_node.name] = mcp_tool_node
                self.mcp_server_info[mcp_server]["tools"][mcp_tool_node.name] = mcp_tool_node

    def _setup_tools(self, tools):
        self.tools = {}
        self.tools.update({tool.name: tool for tool in self.basic_tools})
        if tools:
            assert all(tool.name and tool.description for tool in tools), (
                "All tools need both a name and a description!"
            )
            self.tools.update({tool.name: tool for tool in tools})
        # setup essential tools FinalAnswerTool
        from hie_agent.tools.default_tools import FinalAnswerTool
        self.tools[FinalAnswerTool.name] = FinalAnswerTool()

    def _setup_sub_agents(self, sub_agents):
        self.sub_agents = {}
        self.sub_agents.update({agent.name: agent for agent in self.basic_sub_agents})
        if sub_agents:
            assert all(agent.name and agent.description for agent in sub_agents), (
                "All sub agents need both a name and a description!"
            )
            self.sub_agents.update({agent.name: agent for agent in sub_agents})
    
    def _setup_allowed_libraries(self, allowed_libraries):
        self.allowed_libraries = self.default_libraries + allowed_libraries
    
    def _setup_temp_storage(self):
        self.temp_storage = os.path.join(TEMP_STORAGE_PATH, self.name)
        if not os.path.exists(self.temp_storage):
            os.makedirs(self.temp_storage)
    
    def _preprocess_task(self, task):
        if self.config["react_step_limit"]:
            react_step_limit = self.config["react_step_limit"]
            react_step_limit_footer = f"IMPORTANT: You must complete the given task (derive the final_answer) within {react_step_limit} or fewer though-code steps."
            task = f"{task}\n\n{react_step_limit_footer}"

        return task

    def current_time(self) -> str:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return current_time
    
    #@print_process(option["print_process"])
    def _init_planning_prompt(self, task) -> str: 
        initial_plan_prompt = self._populate_template(
            template=self.prompt_templates["planning_prompt"]["initial_plan"],
            variables={
                "tools": self.tools,
                "sub_agents": self.sub_agents,
                "mcp_servers": self.mcp_server_info,
                "task": task,
                "current_time": self.current_time(),
            },
        )
        return initial_plan_prompt
    
    #@print_process(option["print_process"])
    def _init_system_prompt(self) -> str: 
        system_prompt = self._populate_template(
            template=self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "sub_agents": self.sub_agents,
                "mcp_servers": self.mcp_server_info,
                "allowed_libraries": ( 
                    "You can import from any package you want."
                    if "*" in self.allowed_libraries
                    else str(self.allowed_libraries)
                ),
                "is_main_agent": self.is_main_agent,
                "is_sub_agent": not self.is_main_agent,
                "final_result_storage_path": FINAL_RESULT_STORAGE_PATH,
                "temp_storage_path": self.temp_storage,
                "current_time": self.current_time(),
            },
        )
        return system_prompt
    
    # def _init_subagent_task_prompt(self, task) -> str:
    #     subagent_task_prompt = self._populate_template(
    #         template=self.prompt_templates["task"]["sub_agent"],
    #         variables={
    #             "name": self.name,
    #             "description": self.description,
    #             "task": task,
    #         },
    #     )
    #     return subagent_task_prompt
    
    #@print_process(option["print_process"])
    def _init_task_prompt(self, task): 
        if self.is_main_agent:
            task_prompt = self._populate_template(
                template=self.prompt_templates["task"]["main_agent"],
                variables={
                    "task": task,
                },
            )
        else:
            task_prompt = self._populate_template(
                template=self.prompt_templates["task"]["sub_agent"],
                variables={
                    "name": self.name,
                    "description": self.description,
                    "task": task,
                },
            )
        # task_prompt = f"Task:\n{task}"
        return task_prompt

    def _populate_template(self, template: str, variables: Dict[str, Any]) -> str:
        compiled_template = Template(template, undefined=StrictUndefined)
        try:
            return compiled_template.render(**variables)
        except Exception as e:
            raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")
        
    def _initial_planning_step(self, task) -> None: # 유저의 피드백을 받아 plan을 수정하는 기능도 넣으면 좋을듯 (구현완료)
        report_planning_header = "Here are the facts I know and the plan of action that I will follow to solve the task:\n"
        report_feedback_header = "Here is a revised plan to reflect your requests.\n"
        feedback_request_footer = "\nIf you have any modifications or additional requests for this plan, please let me know."

        user_feedback_header = "Here are some additional requests and modificaiton to your survey and planning: "
        user_feedback_footer = "IMPORTANT: You must follow the instructions and format for survey and planning that were initially provided when you conduct your research and plan revisions."

        def _iterative_user_feedback(self, planning_result_with_report_planning_header):
            planning_result_with_rp_header_fr_footer = f"{planning_result_with_report_planning_header}{feedback_request_footer}"
            latest_planning_result = planning_result_with_rp_header_fr_footer

            print(f"\n<Planning result modification with user feedback>\n\n[Current planning result]\n{latest_planning_result}")

            while True:
                
                user_feedback = input("\nPlease enter any additional requests for the plan. If none, type 'q' or 'quit'.")
                

                if user_feedback in ("q", "quit"):
                    print("\n- User feedback is finished.\n")
                    # latest_planning_result의 report_feedback_header, feedback_request_footer, report_planning_header 모두 제거하고 report_planning_header 붙여서 내보내기
                    modified_planning_result = latest_planning_result.replace(report_feedback_header, "").replace(feedback_request_footer, "").replace(report_planning_header, "")
                    modified_planning_result = f"{report_planning_header}{modified_planning_result}"
                    return modified_planning_result
                else:
                    user_feedback = f"{user_feedback_header}{user_feedback}\n{user_feedback_footer}"

                    self.plan_memory.add_history("ai", planning_result_with_rp_header_fr_footer)
                    self.plan_memory.add_history("user", user_feedback)
                    planning_result: AIMessage = self.llm_engine(self.plan_memory.history)
                    planning_result_with_rf_header_fr_footer = f"{report_feedback_header}{planning_result.content}{feedback_request_footer}"
                    latest_planning_result = planning_result_with_rf_header_fr_footer

                    print(f"\n<Planning modification complete>\n\nRevised planning result:\n{latest_planning_result}")

        initial_plan_prompt = self._init_planning_prompt(task)
        self.plan_memory.add_initial_plan(initial_plan_prompt)

        # output planning_result
        planning_result: AIMessage = self.llm_engine(self.plan_memory.history)
        planning_result_with_report_planning_header = f"{report_planning_header}{planning_result.content}"

        if not self.config["planning_feedback"]:
            # add planning result to memory
            self.plan_memory.add_initial_planning_result(planning_result_with_report_planning_header)

        else: # user can feedback planning result iteratively.
            modified_planning_result_with_report_planning_header = _iterative_user_feedback(self, planning_result_with_report_planning_header)
            # add planning result to memory
            self.plan_memory.add_initial_planning_result(modified_planning_result_with_report_planning_header)
    
    def _initial_react_step(self, task, planning_result): # make intial react process sequence: 1. system prompt 2. user input task, 3. planning result of ai
        # init system prompt
        self.system_prompt = self._init_system_prompt()
        self.react_memory.add_system_prompt(self.system_prompt)

        # init task prompt
        # self.task_prompt = self._init_task_prompt(task)
        self.task_prompt = task
        self.react_memory.add_task_prompt(self.task_prompt)

        # add planning result to memory
        self.react_memory.add_history("ai", planning_result)
        return None
    
    def _correct_code_format(self, thought_action): # TODO FUTURE생성된 코드의 포맷이 잘못된 경우 교정. 추후 구현 예정. 지금은 그냥 바이패스
        return thought_action
    
    def _parse_code(self, thought_action: str) -> str | None:
        code_start_marker = "Code:\n```py"
        code_end_marker = "```"

        start_index = thought_action.find(code_start_marker)
        if start_index == -1:
            return None

        start_index += len(code_start_marker)
        end_index = thought_action.find(code_end_marker, start_index)
        if end_index == -1:
            return None

        parsed_code = thought_action[start_index:end_index].strip()
        return parsed_code

    def _format_observation(self, observation: str) -> str:
        formatted_observation = f"Observation:\n {observation}"
        return formatted_observation
    
    def _format_final_answer(self, final_results):
        final_answer = f"Here is the final answer of given task: {final_results}"
        return final_answer
    
    async def __call__(self, task):
        # IO stream 이슈 해결을 위한 코드
        upper_out = sys.stdout
        sys.stdout = self.code_executor._original_stdout

        # notify agent operation
        print(fill_doubleline_with_string(f"{self.name} Started"))
        task = self._init_task_prompt(task)
        task = self._preprocess_task(task)
        print(f"\n<<Task Received>>\n\nGiven task:\n{task}")

        # initial planning step
        print("\n<<Start Initial Planning Step>>")
        print(f"- Agent is making a plan to solve the given task...")
        self._initial_planning_step(task)
        print("\n<<Finish Initial Planning Step>>\n")
        print(f"Final initial planning result:\n{self.plan_memory.initial_planning_result.content}")

        # ReAct step
        # ReAct initial step
        self._initial_react_step(task, self.plan_memory.initial_planning_result.content)
        # ReAct loop
        print(f"\n<<Start ReAct Loop of {self.name}>>")
        while True:
            print(fill_line_with_string(f"ReAct Step {self.react_step} of {self.name}"))

            # 1. thought and action
            thought_action: AIMessage = self.llm_engine(self.react_memory.history) # ? 허용된 라이브러리 말고도 자꾸 다른 라이브러리 임포트 하는 문제가 끔 발생
            self.react_memory.add_history("ai", thought_action.content)

            print(f"<Derived thought and action>\n{thought_action.content}\n")

            # parse code
            code_block: str = self._parse_code(thought_action.content)
            if not code_block: # if code format is broken, fix code format and try parsing code again
                corrected_thought_action = self._correct_code_format(thought_action.content)
                code_block = self._parse_code(corrected_thought_action)
            
            # 2. execute code and get observation.
            if code_block is None: # 코드 파싱 실패해서 아무런 코드가 없을때
                observation = """No code was successfully parsed. Therefore, no code was executed. The code block you derived is probably formatted incorrectly. Please follow the code block format of the system prompt."""
                is_final_answer = False
            else:
                observation, is_final_answer = await self.code_executor.execute(code_block)
            formatted_observation = self._format_observation(observation)
            self.react_memory.add_history("user", formatted_observation)

            # Check if the current step has exceeded the maximum step.
            if self.react_step > self.config["max_react_step"]:
                if self.is_main_agent:
                    self.tools['final_answer'].final_results = f"The number of executions of main agent has exceeded the maximum number of executions. You need to provide simpler and more compressed tasks to the managed agent so that the agent can complete the given task in fewer attempts. Or, increase the maximum number of executions by adjusting max_react_step of config.py. Here is Execution log of main agent.\n[Execution Log]\n{self.react_memory.provide_input()}"
                else:
                    self.tools['final_answer'].final_results = f"The number of executions of your managed agent '{self.name}' has exceeded the maximum number of executions. You need to provide simpler and more compressed tasks to the managed agent so that the agent can complete the given task in fewer attempts. Another reason for the number of executions to exceed could be that managed agent '{self.name}' failed or got stuck in solving the task so you should consider solving the problem without using this managed agent. Here is the execution log of managed agent '{self.name}.\n[Execution Log]\n{self.react_memory.provide_input()}"

                print(f"***Max ReAct Step Exceeded***\n{self.name} has exceeded the maximum number of react steps. It is forcibly terminated.")
                is_final_answer = True
                
            # 3. check if this step output final answer.
            if not is_final_answer:
                print(f"<Observation Result>\n{formatted_observation}\n")
                self.react_step += 1
            else: 
                if self.is_main_agent:
                    print(f""" 
<Final Answer Received>
- Hierarchical agent system complete your request task.
[Final Results]
{self.tools['final_answer'].final_results}""")
                else:
                    print(f"""
<Final Answer Received>
- This agent resulted final answer. This final answer will be sent to the upper agent.
Final answer result of {self.name}:
{self.tools['final_answer'].final_results}\n""")
                break
        
        print(f"\n<<Finish ReAct Loop of {self.name}>>\n")
        print(fill_doubleline_with_string(f"{self.name} Finished"))
        # clear memory of code_executor, plan_memory and react_memory and react_step when react loop ends (final answer is called.).
        self.code_executor.clear_global_variables()
        self.plan_memory.clear_memory()
        self.react_memory.clear_memory()
        self.react_step = 1
        sys.stdout = upper_out

        final_answer: str = self._format_final_answer(self.tools['final_answer'].final_results)
        return final_answer

# class SubAgentNode(MainAgentNode):
#     def __init__(self, name: str, description: str, inputs: List[InputFormatContainer], outputs: List[OutputFormatContainer], tools: List['ToolNode'], sub_agents: List['SubAgentNode']):
#         super().__init__(name, description, inputs, outputs, tools, sub_agents)

class SubAgentNode(MainAgentNode):
    name: str = "sub_agent"
    description: str
    config: Dict[str, Any] = SUB_AGENT_DEAFULT_CONFIG
    basic_tools: List['ToolNode'] = {}
    basic_sub_agents: List['SubAgentNode'] = {}
    default_libraries = DEFAULT_LIBRARIES
    is_main_agent = False

    def __init__(self, name: str="", description: str="", config: Dict[str, Any]=None):
        super().__init__(config=config)
        # set subagent info
        if name:
            self.name = name
        if description:
            self.description = description
    
    @classmethod
    async def create(cls, name: str="", description:str="", tools: List['ToolNode']=[], sub_agents: List['SubAgentNode']=[], mcp_servers: List[str]=[], async_exit_stack: AsyncExitStack=None, allowed_libraries: List[str]=[], config: Dict[str, Any]=None):
        agent_ins = cls(name, description, config)

        await agent_ins._perform_async_init(tools=tools, sub_agents=sub_agents, mcp_servers=mcp_servers, async_exit_stack=async_exit_stack, allowed_libraries=allowed_libraries, config=config)

        return agent_ins
    
    def _format_final_answer(self, final_results):
        final_answer = f"Here is the final answer from your managed agent {self.name}: {final_results}"
        return final_answer

    async def __call__(self, task):
        # IO stream 이슈 해결을 위한 코드
        upper_out = sys.stdout
        sys.stdout = self.code_executor._original_stdout

        # notify agent operation
        print(fill_doubleline_with_string(f"{self.name} Started"))
        task = self._init_task_prompt(task)
        task = self._preprocess_task(task)
        print(f"\n<<Task Received>>\n\nGiven task:\n{task}")

        # initial planning step
        print("\n<<Start Initial Planning Step>>")
        print(f"- Agent is making a plan to solve the given task...")
        self._initial_planning_step(task)
        print("\n<<Finish Initial Planning Step>>\n")
        print(f"Final initial planning result:\n{self.plan_memory.initial_planning_result.content}")

        # ReAct step
        # ReAct initial step
        self._initial_react_step(task, self.plan_memory.initial_planning_result.content)
        # ReAct loop
        print(f"\n<<Start ReAct Loop of {self.name}>>")
        while True:
            

            print(fill_line_with_string(f"ReAct Step {self.react_step} of {self.name}"))

            # 1. thought and action
            thought_action: AIMessage = self.llm_engine(self.react_memory.history) # ? 허용된 라이브러리 말고도 자꾸 다른 라이브러리 임포트 하는 문제 가끔 발생
            self.react_memory.add_history("ai", thought_action.content)

            print(f"<Derived thought and action>\n{thought_action.content}\n")

            # parse code
            code_block: str = self._parse_code(thought_action.content)
            if not code_block: # if code format is broken, fix code format and try parsing code again
                corrected_thought_action = self._correct_code_format(thought_action.content)
                code_block = self._parse_code(corrected_thought_action)
            
            # 2. execute code and get observation.
            if code_block is None: # 코드 파싱 실패해서 아무런 코드가 없을때
                observation = """No code was successfully parsed. Therefore, no code was executed. The code block you derived is probably formatted incorrectly. Please follow the code block format of the system prompt."""
                is_final_answer = False
            else:
                observation, is_final_answer = await self.code_executor.execute(code_block)
            formatted_observation = self._format_observation(observation)
            self.react_memory.add_history("user", formatted_observation)

            # Check if the current step has exceeded the maximum step.
            if self.react_step > self.config["max_react_step"]:
                if self.is_main_agent:
                    self.tools['final_answer'].final_results = f"The number of executions of main agent has exceeded the maximum number of executions. You need to provide simpler and more compressed tasks to the managed agent so that the agent can complete the given task in fewer attempts. Or, increase the maximum number of executions by adjusting max_react_step of config.py. Here is Execution log of main agent.\n[Execution Log]\n{self.react_memory.provide_input()}"
                else:
                    self.tools['final_answer'].final_results = f"The number of executions of your managed agent '{self.name}' has exceeded the maximum number of executions. You need to provide simpler and more compressed tasks to the managed agent so that the agent can complete the given task in fewer attempts. Another reason for the number of executions to exceed could be that managed agent '{self.name}' failed or got stuck in solving the task so you should consider solving the problem without using this managed agent. Here is the execution log of managed agent '{self.name}.\n[Execution Log]\n{self.react_memory.provide_input()}"

                print(f"***Max ReAct Step Exceeded***\n{self.name} has exceeded the maximum number of react steps. It is forcibly terminated.")

                is_final_answer = True
                
            # 3. check if this step output final answer.
            if not is_final_answer:
                print(f"<Observation Result>\n{formatted_observation}\n")
                self.react_step += 1
            else: 
                if self.is_main_agent:
                    print(f""" 
<Final Answer Received>
- Hierarchical agent system complete your request task.
[Final Results]
{self.tools['final_answer'].final_results}""")
                else:
                    print(f"""
<Final Answer Received>
- This agent resulted final answer. This final answer will be sent to the upper agent.
Final answer result of {self.name}:
{self.tools['final_answer'].final_results}\n""")
                break
        
        print(f"\n<<Finish ReAct Loop of {self.name}>>\n")
        print(fill_doubleline_with_string(f"{self.name} Finished"))
        # clear memory of code_executor, plan_memory and react_memory and react_step when react loop ends (final answer is called.).
        self.code_executor.clear_global_variables()
        self.plan_memory.clear_memory()
        self.react_memory.clear_memory()
        self.react_step = 1
        sys.stdout = upper_out

        final_answer: str = self._format_final_answer(self.tools['final_answer'].final_results)
        return final_answer

if __name__ == "__main__":
    # main_agent = MainAgentNode()
    # final_result = main_agent("Make a simple guide to tourism in France using web search.")
    
    sub_agent = SubAgentNode(name="search_agent", description="This agent is a search agent, which searches the web to find all information about the input task.")
    sub_agent("Make a simple guide to tourism in France.")


