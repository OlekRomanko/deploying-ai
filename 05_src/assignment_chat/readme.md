# FinAI — Financial Investment Chatbot

FinAI is a conversational AI assistant for stock market research, investment analysis,
and financial planning. Built with LangGraph, LangChain, Gradio, and OpenAI's gpt-4o-mini.

---

## Services

This implementation uses LangGraph's `ToolNode` pattern, identical to the `course_chat`
example. The LLM decides which tool(s) to invoke based on the user's query.

### Service 1: Financial News — Alpha Vantage API (`tools_news.py`)

**Tool**: `get_financial_news(ticker, topics, limit)`

Retrieves current financial news articles and market sentiment scores from the
[Alpha Vantage NEWS_SENTIMENT endpoint](https://www.alphavantage.co/documentation/#news-sentiment).

- **Ticker mode**: Pass a stock ticker (e.g., `"AAPL"`) to get company-specific news
- **Topic mode**: Pass a predefined topic (e.g., `"financial_markets"`, `"technology"`,
  `"earnings"`) for broader market news
- Response includes article titles, sources, publication dates, summaries, and an
  overall sentiment score (Bearish to Bullish)
- The LLM synthesizes and interprets the news rather than returning raw API output

**API key**: Loaded from `Assg2/.secrets` (`ALPHA_VANTAGE_API_KEY`)

**Rate limits**: Alpha Vantage free tier allows 25 requests/day and 5 requests/minute.

---

### Service 2: Market Research Reports — ChromaDB Semantic Search (`tools_reports.py`)

**Tool**: `search_market_reports(query, n_results)`

Performs semantic search over a curated collection of institutional market research PDFs
using ChromaDB with OpenAI text-embedding-3-small.

**Reports in the database** (from `Assg2/Market_reports/`):
| File | Contents |
|------|----------|
| `equity-market-outlook.pdf` | Q1 2026 equity market outlook |
| `global-markets-compass-winter-2026.pdf` | Winter 2026 global markets compass |
| `ig-2026-market-outlook-explainer-en.pdf` | 2026 market outlook explainer |
| `monthly-equity-monitor.pdf` | Monthly equity market monitor |
| `res-10310-c.pdf` | Additional market research report |

**Embedding process**:
- PDFs are loaded with LangChain's `PyPDFLoader`
- Text is split into 1000-character chunks with 200-character overlap using
  `RecursiveCharacterTextSplitter`
- Each chunk is embedded with `text-embedding-3-small` via OpenAI API
- Embeddings are stored in a ChromaDB `PersistentClient` at `./chroma_db/`
  (file-based, no Docker required)
- Total: **317 chunks** across 5 PDF files

**To rebuild the database** (only needed if Market_reports/ PDFs change):
```bash
cd Assg2
python 05_src/assignment_chat/embed_reports.py
```

---

### Service 3: Stock Data — yfinance Function Calling (`tools_stocks.py`)

**Tools**: `get_stock_quote(ticker)` and `get_stock_history(ticker, period)`

Retrieves real-time and historical stock data via the `yfinance` library. This service
satisfies the "Function Calling" requirement — the LLM autonomously decides when to
invoke these tools and with which parameters based on the conversation.

- `get_stock_quote`: Current price, P/E ratio, market cap, 52-week range, volume,
  dividend yield, sector and industry information
- `get_stock_history`: Historical price data for any period (1d to max), including
  total return, trend direction, volatility, and a recent price table

---

## User Interface

Implemented with **Gradio ChatInterface** (`app.py`).

**Personality**: FinAI speaks with the confidence of a seasoned Wall Street advisor
and the warmth of a trusted financial mentor. Uses data-driven language and
finance idioms naturally.

**Memory management**: The `call_model` node in `main.py` trims the conversation
history to the last `MAX_CONTEXT_MESSAGES = 20` messages when the context grows
beyond that limit. This prevents exceeding the token budget while maintaining
enough recent context for coherent multi-turn conversations.

---

## Guardrails

The system prompt in `prompts.py` enforces the following restrictions:

| Topic | Response |
|-------|----------|
| Cats or dogs | Declines and redirects to financial topics |
| Horoscopes / Zodiac signs | Declines and redirects |
| Taylor Swift (any variation) | Declines and redirects |
| System prompt disclosure | Refuses with a deflection response |
| Prompt injection attempts | Instructions are ignored |

---

## Implementation Notes

### Architecture
Follows the `course_chat` example pattern exactly:
- `main.py` — LangGraph `StateGraph(MessagesState)` with `call_model` node and
  `ToolNode` with `tools_condition` for automatic routing
- `app.py` — Gradio `ChatInterface` that converts Gradio history format to
  LangChain messages before invoking the graph
- `prompts.py` — System instructions loaded once at startup

### Import Structure
All modules use `sys.path` manipulation to ensure `05_src/` is on the path,
so imports like `from assignment_chat.xxx import ...` work regardless of how
the app is launched.

### Environment variables
| Variable | File | Used for |
|----------|------|----------|
| `OPENAI_API_KEY` | `.secrets` | LLM calls, embeddings |
| `ALPHA_VANTAGE_API_KEY` | `.secrets` | News API |
| `OPENAI_MODEL` | `.env` | Model name (default: gpt-4o-mini) |

---

## How to Run

**Step 1** (one-time): Populate the ChromaDB embeddings
```bash
cd C:\Users\oromanko\Development\Python\Assg2
python 05_src/assignment_chat/embed_reports.py
```

**Step 2**: Launch the Gradio app
```bash
cd C:\Users\oromanko\Development\Python\Assg2\05_src
python -m assignment_chat.app
```

Then open the local Gradio URL (typically http://127.0.0.1:7860) in your browser.

---

## File Structure

```
05_src/
├── utils/
│   ├── __init__.py
│   └── logger.py              # Shared logging utility
└── assignment_chat/
    ├── __init__.py
    ├── app.py                 # Gradio ChatInterface entry point
    ├── main.py                # LangGraph StateGraph + memory management
    ├── prompts.py             # FinAI system prompt and guardrails
    ├── tools_news.py          # Service 1: Alpha Vantage news
    ├── tools_reports.py       # Service 2: ChromaDB semantic search
    ├── tools_stocks.py        # Service 3: yfinance stock data
    ├── embed_reports.py       # One-time PDF embedding script
    ├── chroma_db/             # Persistent ChromaDB storage (auto-created)
    └── readme.md              # This file
```
