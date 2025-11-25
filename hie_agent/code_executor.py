import builtins
import sys
import io
import importlib
import traceback

from typing import *

class LocalPythonExecutor:
    def __init__(self, allowed_libraries = None, tools = None, mcp_tools = None, sub_agents =None):
        # 기본 허용된 라이브러리 목록 (없으면 빈 리스트)
        self.allowed_libraries = allowed_libraries if allowed_libraries else []
        # 툴 및 하위 에이전트 임포트
        self.imported_variables = {}
        self._import_tools(tools)
        self._import_mcp_tools(mcp_tools)
        self._import_subagents(sub_agents)

         # 전역 변수 저장 공간
        self.global_variables = {}
        self.global_variables.update(self.imported_variables)  # 외부 함수 및 인스턴스 추가

        # 출력 캡처를 위한 설정
        self._original_stdout = sys.stdout

    async def execute(self, code: str):
        def _wrap_sync_func_to_async_func(code_string: str, wrapper_func_name: str = "_dynamic_async_wrapper", initial_local_vars: str = "_initial_local_vars", final_local_vars: str = "_final_local_vars") -> str:
            import textwrap
            # 코드 문자열의 각 줄에 들여쓰기 추가 (async def 안에 들어가도록)
            indented_code_string = textwrap.indent(code_string, "    ") # 공백 4개로 들여쓰기

            wrapped_code = f"""
import asyncio # 코드 문자열 내에서 asyncio를 사용할 경우 필요 (보통 LLM이 알아서 넣어줌)
import inspect # 필터링 로직을 위해 래퍼 함수 내부에 필요

async def {wrapper_func_name}():
# 래퍼 함수 시작 시 지역 변수 스냅샷 (필터링 기준)
    {initial_local_vars} = locals().copy()

    # 래퍼 함수의 매개변수(없지만 혹시 있다면)와 snapshot 변수는 필터링 대상
    _wrapper_internal_names = set({initial_local_vars}.keys()) | set(['{initial_local_vars}', '_wrapper_internal_names']) # 필터링 제외 목록

{indented_code_string}

    # 래퍼 함수 종료 직전 지역 변수 스냅샷
    _final_locals_snapshot = locals().copy()

    _variables_to_export = {{}}
    for name, value in _final_locals_snapshot.items():
        # 내부 변수 이름 목록에 없거나, '__'로 시작하지 않는 변수들 위주로 검토
        if name not in _wrapper_internal_names and not name.startswith('__'):
             # 초기 스냅샷에 이름이 없는 경우 -> 새로 생성된 변수
            if name not in {initial_local_vars}:
                _variables_to_export[name] = value
            # 초기 스냅샷에 이름은 있지만 값이 달라진 경우 -> 기존 변수의 값 변경
            # (간단히 객체 ID가 달라진 경우로 판단하거나 값을 비교)
            # 객체 ID 비교는 새 객체 할당만 감지, 값 비교는 내용 변경 감지 (복잡)
            # 여기서는 이름을 기준으로 새로 생겼거나 변경된 것 추정 (휴리스틱)
            # 보다 엄밀하려면 각 변수의 이전/이후 값 비교 로직 추가 필요
            # 예: if name in initial_locals_snapshot and initial_locals_snapshot[name] is not value: ...
            # 여기서는 이름만으로 새 변수인지 판단하는 간략화된 필터 사용
            pass # 기본 필터링 로직은 아래 return 문에서 처리

            # 최종 필터링: 내부 변수 제외하고 반환
    return {{name: value for name, value in _variables_to_export.items()}} # 휴리스틱 필터링
"""
            return wrapped_code
        
        
        wrapper_func_name = "_dynamic_async_wrapper"
        initial_local_vars = "_initial_local_vars"
        final_local_vars = "_final_local_vars"
        

        ## 생성된 코드 문자열을 비동기 함수로 wrapping
        wrapped_code_string = _wrap_sync_func_to_async_func(code, wrapper_func_name, initial_local_vars, final_local_vars)

        # 코드에 필요한 라이브러리 import 처리 (라이브러리만 검사)
        self._check_library_imports(code)

        # 출력 캡처 초기화
        output = io.StringIO()
        sys.stdout = output

        # 비동기 함수 선언
        try:
            exec(wrapped_code_string, self.global_variables)
        except Exception as e:
            # exc_type, exc_value, exc_traceback = sys.exc_info()
            full_error_message = str(traceback.format_exc())
            error_message = '\n'.join(full_error_message.strip().split('\n')[3:])
            print(f"\n[Error Occured]\nAn error occurred while running this code. \nError message:\n'''\n{error_message}\n'''")

        # 비동기화된 코드 실행
        try:
            async_func_to_run = self.global_variables[wrapper_func_name] # 선언된 비동기 함수 
            returned_local_vars = await async_func_to_run()  # 비동기 함수 실행
            self.global_variables.update(returned_local_vars)  # 반환된 지역 변수 업데이트

        except Exception as e:
            full_error_message = str(traceback.format_exc())
            error_message = '\n'.join(full_error_message.strip().split('\n')[3:])
            print(f"\n[Error Occured]\nAn error occurred while running this code. \nError message:\n'''\n{error_message}\n'''")
        
        finally:
            if wrapper_func_name in self.global_variables:
                del self.global_variables[wrapper_func_name]
            # if '__builtins__' in self.global_variables: # exec 추가 시 __builtins__가 추가될 수 있음
            #     del self.global_variables['__builtins__']


        # 실행 결과를 캡처한 출력 반환
        result = output.getvalue()
        sys.stdout = self._original_stdout  # 원래 stdout으로 복구

        if 'final_answer(' in code: # TODO: 에러코드를 observation으로 출력하도록, 이거 예외처리 너무 원시적임... 
            is_final_answer = True
        else:
            is_final_answer = False

        return result, is_final_answer

    def _import_tools(self, tools):
        """
        툴들을 전역 변수에 등록
        """
        self.tools = tools if tools else {}
        for tool_name, tool_ins in self.tools.items():
            self.imported_variables[tool_name] = tool_ins
    
    def _import_mcp_tools(self, mcp_tools):
        """
        mcp 툴들을 전역 변수에 등록
        """
        self.mcp_tools = mcp_tools if mcp_tools else {}
        for mcp_tool_name, mcp_tool_ins in self.mcp_tools.items():
            self.imported_variables[mcp_tool_name] = mcp_tool_ins

    def _import_subagents(self, sub_agents):
        """
        하위 에이전트들을 전역 변수에 등록
        """
        self.sub_agents = sub_agents if sub_agents else {}
        for sub_agent_name, sub_agent_ins in self.sub_agents.items():
            self.imported_variables[sub_agent_name] = sub_agent_ins

    def _check_library_imports(self, code: str): # TODO FUTURE 만약 config에서 허용된 라이브러리 목록을 라이브러리 뿐만 아니라 from x import y의 x와 y를 모두 지정하게 된다면, x와 y를 모두 검사하도록 기능 업데이트 필요, 현재는 생성된 코드에 from x import y가 있다면 x가 허용된 라이브러리 목록과 일치하는 지만 검사
        """
        코드에서 사용하는 라이브러리를 확인하고, 허용된 라이브러리만 import
        """
        # 코드에서 import 문 찾기
        imports = [line for line in code.splitlines() if line.strip().startswith('import') or line.strip().startswith('from')]
        for import_line in imports:
            if 'import' in import_line:
                # 'import' 라인 처리
                lib_name = import_line.split()[1].split('.')[0]
            elif 'from' in import_line:
                # 'from' 라인 처리
                lib_name = import_line.split()[1].split('.')[0]
            
            # 라이브러리가 허용 목록에 없는 경우 경고
            if lib_name not in self.allowed_libraries and lib_name not in builtins.__dict__:
                print(f"경고: '{lib_name}' 라이브러리는 허용되지 않은 라이브러리입니다.")
                continue

            # 허용되지않은 라이브러리는 경고 출력 (막지는 않음)
            if lib_name not in self.global_variables:
                try:
                    self.global_variables[lib_name] = importlib.import_module(lib_name)
                except ModuleNotFoundError:
                    print(f"경고: '{lib_name}' 라이브러리를 찾을 수 없습니다.")

    def get_output(self):
        """ 최근 실행된 코드의 출력 결과를 반환 """
        return self.output.getvalue()

    def clear_output(self):
        """ 출력 결과 초기화 """
        self.output.truncate(0)
        self.output.seek(0)
    
    def clear_global_variables(self):
        """ 전역 변수 초기화 """
        self.global_variables.clear()
        self.global_variables.update(self.imported_variables)

# 예시 사용법
if __name__ == "__main__":
    # 사용 가능한 라이브러리 목록
    allowed_libraries = ['math', 'os']

    # 외부 함수 정의 예시
    def add(a, b):
        return a + b

    # 외부 함수 인스턴스 정의 예시
    external_functions = {
        'calculator': add
    }

    # PythonExecutor 객체 생성
    executor = LocalPythonExecutor(allowed_libraries, external_functions)

    # 첫 번째 코드 실행
    code_1 = """
import math
print(math.sqrt(16))
print("hello_world")
a = 1
b = 2
"""
    result = executor.execute(code_1)  # 출력: 4.0
    print(result)
    print("hi")
    # 두 번째 코드 실행
    code_2 = """
result = calculator(a, b)
print(result)
"""
    print(executor.execute(code_2))  # 출력: 15
    a = 1
