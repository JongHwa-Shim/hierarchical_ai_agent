from hie_agent.nodes import SubAgentNode
from .config import *
from .browser import SimpleTextBrowser
from .tools import *
from ...tools.custom_tools import GoogleSearchTool

class BrowserAgent(SubAgentNode):
    name = "browser_agent"
    description = """A managed agent that will search the internet to answer your question.
Ask him for all your questions that require Complex and detailed browsing the web.
Provide him as much context as possible, in particular if you need to search on a specific timeframe!
And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords."""

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    basic_tools = [
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        GoogleSearchTool()
    ]