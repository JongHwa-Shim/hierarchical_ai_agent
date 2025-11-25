from typing import TypedDict, List, Dict, Any, Optional, Union, Dict, Tuple, Callable, TypeVar, Generic
import os
# class InputFormatContainer(TypedDict): # deprecated
#     def __init__(self, name: str, description: str, type: str):
#         self.name = name
#         self.description = description
#         self.type = type

# class OutputFormatContainer(TypedDict): # deprecated
#     def __init__(self, description: str, type: str):
#         self.description = description
#         self.type = type

class Memory():
    def __init__(self):
        self.history = []

    def add_history(self, message_type, message):
        if message_type == "system":
            self.history.append(SystemMessage(message))
        elif message_type == "user":
            self.history.append(UserMessage(message))
        elif message_type == "ai":
            self.history.append(AIMessage(message))
        else:
            raise ValueError("Invalid message type. Must be 'system', 'user', or 'ai'.")
    
    def provide_input(self):
        output = ""
        for message in self.history:
            output += f"{message}\n"
        return output
    
class PlanMemory(Memory):
    def __init__(self):
        self.initial_planning_prompt: str
        self.initial_planning_result: str
        self.history = []

    def add_initial_plan(self, initial_plan: str):
        self.initial_planning_prompt = UserMessage(initial_plan)
        self.history.append(self.initial_planning_prompt)
    
    def add_initial_planning_result(self, planning_result: str):
        self.initial_planning_result = AIMessage(planning_result)
        self.history.append(self.initial_planning_result)

    def clear_memory(self):
        self.initial_planning_prompt = None
        self.initial_planning_result = None
        self.history = []

class ReActMemory(Memory):
    def __init__(self):
        self.system_message = ""
        self.task_message= ""
        self.history = []

    def add_system_prompt(self, system_message: str):
        self.system_message = SystemMessage(system_message)
        self.history.append(self.system_message)
    
    def add_task_prompt(self, task_message: str):
        self.task_message = UserMessage(task_message)
        self.history.append(self.task_message)

    def clear_memory(self):
        self.system_message = None
        self.task_message = None
        self.history = []

class MessageType():
    speaker: str
    content: str
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return f"{self.speaker}: {self.content}"
    
    def append_content(self, content: str):
        self.content = self.content + r"\n" + content

class AIMessage(MessageType):
    speaker = "AI"

class UserMessage(MessageType):
    speaker = "User"

class SystemMessage(MessageType):
    speaker = "System"

class LLMContainer():
    def __init__(self, *args, **kwargs): 
        self.config = kwargs
        self.llm_lib = self.config["llm_lib"]
        self.llm_name = self.config["llm_name"]

        if self.llm_lib == "langchain":
            from langchain_openai import ChatOpenAI
            from langchain.schema.output_parser import StrOutputParser

            if self.llm_name == "gpt-4o":
                llm = ChatOpenAI(model=self.llm_name, api_key=self.config.get("llm_api_key"), temperature=self.config.get("temperature"))
            elif self.llm_name == "o1":
                llm = ChatOpenAI(model=self.llm_name, api_key=self.config.get("llm_api_key"), reasoning_effort=self.config.get("reasoning"))
            elif self.llm_name == "gpt-5.1":
                llm = ChatOpenAI(model=self.llm_name, api_key=self.config.get("llm_api_key"), reasoning_effort=self.config.get("reasoning"))
            elif self.llm_name == "gpt-5-mini":
                llm = ChatOpenAI(model=self.llm_name, api_key=self.config.get("llm_api_key"), reasoning_effort=self.config.get("reasoning"))
            elif self.llm_name == "gpt-5-nano":
                llm = ChatOpenAI(model=self.llm_name, api_key=self.config.get("llm_api_key"), reasoning_effort=self.config.get("reasoning"))
                
            parser = self.config.get("parser", StrOutputParser())
            chain =  llm | parser
            self.llm = chain
        else:
            pass # TODO 다양한 프레임웍의 LLM을 지원하기 위해 인스턴스화 시점에 LLM을 결정할 수 있도록 추가 예정
            

    def __call__(self, input):
        processed_input = self._process_input(input)
        if self.llm_lib == "langchain":
            output = self.llm.invoke(processed_input)
            return AIMessage(output)

    def _process_input(self, input): 
        if self.llm_lib == "langchain":
            from langchain_core.messages.system import SystemMessage as LangchainSystemMessage
            from langchain_core.messages.human import HumanMessage as LangchainHumanMessage
            from langchain_core.messages.ai import AIMessage as LangchainAIMessage
            input: List[MessageType]

            message_list = []
            for message in input:
                if isinstance(message, SystemMessage):
                    message_list.append(LangchainSystemMessage(message.content))
                elif isinstance(message, UserMessage):
                    message_list.append(LangchainHumanMessage(message.content))
                elif isinstance(message, AIMessage):
                    message_list.append(LangchainAIMessage(message.content))
            return message_list
        else:
            pass # TODO 다양한 프레임웍의 LLM을 지원하기 위해 인스턴스화 시점에 전처리 알고리즘을 결정할 수 있도록 추가 예정

    
if __name__ == "__main__" : # for test
    llm = LLMContainer(llm_lib="langchain", llm_name="gpt-4o", temperature=0.7, llm_api_key=os.getenv("LLM_API_KEY"))
    chat_history = [
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is the capital of France?"),
        AIMessage("OK, I will answer you questions.")
    ]
    llm(chat_history)
    a = 1
