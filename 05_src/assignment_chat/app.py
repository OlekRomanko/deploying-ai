"""
FinAI — Financial Investment Chatbot
Gradio ChatInterface

Run from the Assg2/05_src directory:
    python -m assignment_chat.app

Or from the Assg2 root directory:
    python 05_src/assignment_chat/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from assignment_chat.main import get_graph
from assignment_chat.utils.logger import get_logger

_logs = get_logger(__name__)

# Load environment variables
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")
load_dotenv(_root / ".secrets")

# Initialize the LangGraph agent
_logs.info("Initializing FinAI agent...")
llm = get_graph()
_logs.info("FinAI agent ready.")


def financial_chat(message: str, history: list[dict]) -> str:
    """
    Gradio chat function. Converts Gradio message history into LangChain
    message format and invokes the LangGraph agent.

    Args:
        message: The current user message.
        history: List of previous messages in Gradio format
                 (dicts with 'role' and 'content' keys).

    Returns:
        The assistant's response as a string.
    """
    _logs.info(f"User message: {message[:100]}{'...' if len(message) > 100 else ''}")

    # Convert Gradio history to LangChain messages
    langchain_messages = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))

    # Add current user message
    langchain_messages.append(HumanMessage(content=message))

    state = {"messages": langchain_messages}

    try:
        response = llm.invoke(state)
        reply = response["messages"][-1].content
        _logs.debug(f"FinAI reply: {reply[:100]}{'...' if len(reply) > 100 else ''}")
        return reply
    except Exception as e:
        _logs.error(f"Error invoking FinAI agent: {e}")
        return (
            "I encountered an issue processing your request. "
            "Please try again or rephrase your question."
        )


# Gradio ChatInterface configuration
chat = gr.ChatInterface(
    fn=financial_chat,
    title="FinAI — Your Financial Investment Advisor",
    description=(
        "Ask me about stocks, market news, investment strategies, and financial planning. "
        "I have access to live financial news, institutional market research reports, "
        "and real-time stock data."
    ),
    examples=[
        "What is the current price of Apple stock?",
        "What are the latest news and market sentiment for Tesla?",
        "What does the market outlook say about equity markets in 2026?",
        "How has Microsoft stock performed over the last 3 months?",
        "What are the key investment themes for technology in 2026?",
        "Get me recent financial news about the energy sector",
    ],
)

if __name__ == "__main__":
    _logs.info("Starting FinAI Gradio app...")
    chat.launch()
