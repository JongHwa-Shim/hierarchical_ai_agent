from __future__ import annotations
import os
import typing

def print_process(print_process: bool=False): # decorator factory for print result of function
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"<<Start: {func.__name__}>>")
            result = func(*args, **kwargs)
            print(result)
            return result
        if print_process:
            return wrapper
        else:
            return func
    return decorator

def fill_doubleline_with_string(text):
  """터미널 가로폭에 맞춰 문자열을 가운데 정렬하고 양쪽에 '='로 채웁니다.

  Args:
    text: 채울 문자열.

  Returns:
    터미널 가로폭에 맞춰진 문자열. 터미널 폭을 얻을 수 없으면 원본 문자열을 반환합니다.
  """
  try:
    terminal_width = os.get_terminal_size().columns
  except OSError:
    return text  # 터미널 폭을 얻을 수 없으면 원본 문자열 반환

  text_length = len(text)
  padding_length = terminal_width - text_length

  if padding_length <= 0:
    return text  # 채울 공간이 없으면 원본 문자열 반환

  left_padding_length = padding_length // 2
  right_padding_length = padding_length - left_padding_length

  left_padding = "=" * left_padding_length
  right_padding = "=" * right_padding_length

  return f"{left_padding}{text}{right_padding}"

def fill_line_with_string(text):
  """터미널 가로폭에 맞춰 문자열을 가운데 정렬하고 양쪽에 '='로 채웁니다.

  Args:
    text: 채울 문자열.

  Returns:
    터미널 가로폭에 맞춰진 문자열. 터미널 폭을 얻을 수 없으면 원본 문자열을 반환합니다.
  """
  try:
    terminal_width = os.get_terminal_size().columns
  except OSError:
    return text  # 터미널 폭을 얻을 수 없으면 원본 문자열 반환

  text_length = len(text)
  padding_length = terminal_width - text_length

  if padding_length <= 0:
    return text  # 채울 공간이 없으면 원본 문자열 반환

  left_padding_length = padding_length // 2
  right_padding_length = padding_length - left_padding_length

  left_padding = "-" * left_padding_length
  right_padding = "-" * right_padding_length

  return f"{left_padding}{text}{right_padding}"

def print_hie_agent_structure(instance, indent=0, is_last=True, prefix=""):
    """
    클래스 인스턴스의 소유 구조를 재귀적으로 순회하며 출력합니다.
    """
    from hie_agent.nodes import MainAgentNode, SubAgentNode, ToolNode, MCPToolNode
    if indent == 0:
        print(fill_doubleline_with_string("Current AI Agent Hierarchy Structure"))
        print(instance.name)
    else:
        # 마지막 요소가 아니면 '├── ', 마지막 요소면 '└── '
        connector = "└── " if is_last else "├── "
        # 현재 들여쓰기 수준에 맞는 접두사 생성
        print(f"{prefix}{connector}{instance.name}", end="")

        if isinstance(instance, SubAgentNode):
            print(" [sub agent]")
        elif isinstance(instance, ToolNode):
            print(" <native tool>")
        elif isinstance(instance, MCPToolNode):
            print(" <MCP tool>")
        else:
            print("")

    # 다음 들여쓰기를 위한 새로운 접두사 생성
    # 마지막 요소가 아니면 '│   ', 마지막 요소면 '    '
    new_prefix = prefix + ("    " if is_last and indent != 0 else "│   ")

    # Sub_Agent 인스턴스 순회
    if hasattr(instance, 'sub_agents') and instance.sub_agents:
        for i, sub_instance_name in enumerate(instance.sub_agents):
            sub_instance = instance.sub_agents[sub_instance_name] 
            print_hie_agent_structure(sub_instance, indent + 1, i == len(instance.sub_agents) - 1 and not instance.tools, new_prefix)

    # Tool_Agent 인스턴스 순회
    if hasattr(instance, 'tools') and instance.tools:
        for i, tool_instance_name in enumerate(instance.tools):
            tool_instance = instance.tools[tool_instance_name]
            print_hie_agent_structure(tool_instance, indent + 1, i == len(instance.tools) - 1, new_prefix)
    
    if hasattr(instance, 'mcp_tools') and instance.mcp_tools:
        for i, mcp_tool_instance_name in enumerate(instance.mcp_tools):
            mcp_tool_instance = instance.mcp_tools[mcp_tool_instance_name]
            print_hie_agent_structure(mcp_tool_instance, indent + 1, i == len(instance.mcp_tools) - 1, new_prefix)
