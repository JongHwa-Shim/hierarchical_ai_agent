from hie_agent import *
from hie_agent.nodes import ToolNode

# class SubAgentFinalAnswer(ToolNode):
#     """
#     Final answer to the question.
#     """
#     name = 'final_answer'
#     description = "Provides a final answer to the given problem. This tool returns nothing and internally sends the final answer data to your manager. This tool must be called with keyword arguments."
#     input_info = None
#     output_info = None
#     final_results = None

#     def __init__(self, sub_agent_output_info):
#         super().__init__()
#         self.input_info = sub_agent_output_info
#         self.output_info = sub_agent_output_info

#     def _arrange_kwargs(self, kwargs):
#         output_list = []
#         for output in self.output_info:
#             try:
#                 output_list.append(kwargs[output.name])
#             except:
#                 raise KeyError(f"required output '{output.name}' not found in final answer.")
#         return output_list
    
#     def __call__(self, **kwargs):
#         """
#         Call the final answer function.
#         """
#         # self.final_results = FinalAnswer(*args, **kwargs)
#         # return self.final_
#         # results
#         self.final_results = self._arrange_kwargs(kwargs)
#         return self.final_results

class FinalAnswerTool(ToolNode):
    """
    Final answer to the question.
    """
    name = 'final_answer'
    description = "Provides a final answer to the given problem. This tool returns nothing and internally sends the final answer data to your manager."
    input_format = {'answer': {"type": "string", "description": "Provides a final answer to the given problem. This tool returns nothing and internally sends the final answer data to your manager. It should contain all information relevant to the possible problem."}}
    output_format = [{"type": None, "description": "this tool return nothing"}]
    final_results = None
    
    def __call__(self, answer):
        self.final_results = answer