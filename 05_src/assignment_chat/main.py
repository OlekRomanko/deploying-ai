"""
LangGraph StateGraph for FinAI — Financial Investment Chatbot

Architecture mirrors the course_chat example:
- call_model node: LLM decides whether to use a tool or respond directly
- ToolNode: executes whichever tool the LLM selected
- Memory management: trims conversation history when it exceeds MAX_CONTEXT_MESSAGES

Services integrated:
  1. get_financial_news     — Alpha Vantage NEWS_SENTIMENT API
  2. search_market_reports  — ChromaDB semantic search over market research PDFs
  3. get_stock_quote        — yfinance current price and metrics
  4. get_stock_history      — yfinance historical price data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, MessagesState, START
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from assignment_chat.utils.embeddings import _GATEWAY_URL

from assignment_chat.prompts import return_instructions
from assignment_chat.tools_news import get_financial_news
from assignment_chat.tools_reports import search_market_reports
from assignment_chat.tools_stocks import get_stock_quote, get_stock_history
from assignment_chat.utils.logger import get_logger

_logs = get_logger(__name__)

# Load environment variables from Assg2 root
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")
load_dotenv(_root / ".secrets")

# Model and tools configuration
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_tools = [get_financial_news, search_market_reports, get_stock_quote, get_stock_history]
_instructions = return_instructions()

# Memory management: maximum number of messages to keep in context window
MAX_CONTEXT_MESSAGES = 20

chat_agent = ChatOpenAI(
    model=_MODEL,
    openai_api_base=_GATEWAY_URL,
    openai_api_key='any value',
    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
)

_logs.info(f"FinAI initialized with model: {_MODEL}, tools: {[t.name for t in _tools]}")


def call_model(state: MessagesState):
    """LLM node: decides whether to call a tool or generate a final response."""
    messages = state["messages"]

    # Memory management: trim oldest messages if conversation is too long.
    # Always keep the most recent MAX_CONTEXT_MESSAGES messages so the LLM
    # has enough context without exceeding the token budget.
    if len(messages) > MAX_CONTEXT_MESSAGES:
        _logs.debug(
            f"Trimming context: {len(messages)} messages -> {MAX_CONTEXT_MESSAGES}"
        )
        messages = messages[-MAX_CONTEXT_MESSAGES:]

    response = chat_agent.bind_tools(_tools).invoke(
        [SystemMessage(content=_instructions)] + messages
    )
    return {"messages": [response]}


def get_graph():
    """Builds and compiles the LangGraph StateGraph for FinAI."""
    builder = StateGraph(MessagesState)

    builder.add_node(call_model)
    builder.add_node(ToolNode(_tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    graph = builder.compile()
    _logs.info("FinAI LangGraph compiled successfully.")
    return graph
