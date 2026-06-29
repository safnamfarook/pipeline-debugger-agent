# Intelligent Data Pipeline Debugger

An agentic AI system that diagnoses Spark, Airflow, and Kafka pipeline failures using LangGraph, FastAPI, and Groq.

## What It Does

- Accepts pipeline logs via file upload, paste, or REST API
- Uses a ReAct-pattern LangGraph agent with four specialist tools to read, classify, and suggest fixes for pipeline errors
- Maintains conversation memory across turns so follow-up questions work naturally
- Exposes a FastAPI backend with `/debug`, `/debug/log`, `/debug/stream`, and `/health` endpoints
- Fully containerised with Docker for portable deployment

## Architecture

User → Streamlit UI / REST API (FastAPI)
↓
LangGraph Agent (ReAct loop)
↓
┌─────────────────────┐
│       4 Tools       │
├─────────────────────┤
│ read_log_file       │
│ identify_error_type │
│ get_fix_suggestion  │
│ list_available_logs │
└─────────────────────┘
↓
Groq (llama-3.3-70b-versatile)
↓
LangFuse (observability + tracing)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent framework | LangGraph 1.x |
| LLM | Groq — llama-3.3-70b-versatile |
| API layer | FastAPI + Uvicorn |
| UI | Streamlit |
| Observability | LangFuse v3 |
| Containerisation | Docker + Docker Compose |

## Project Structure

Intelligent Data Pipeline Debugger/
├── config.py          # API keys, model settings, paths
├── tools.py           # Four LangGraph tools
├── agent.py           # LangGraph state machine + chat() function
├── app.py             # Streamlit UI
├── api.py             # FastAPI wrapper
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env               # API keys (not committed)
└── example/
└── sample_logs/
├── spark_error.log
└── airflow_error.log

## Quickstart

### Run with Docker (recommended)

```bash
cd "Intelligent Data Pipeline Debugger"

# Add your API keys
cp .env.example .env

# Build and run
docker-compose up --build
```

API available at `http://localhost:8000`  
Docs at `http://localhost:8000/docs`

### Run locally

```bash
pip install -r requirements.txt

# Run the API
python api.py

# Or run the Streamlit UI
streamlit run app.py
```

## API Usage

### Health check
```bash
curl http://localhost:8000/health
```

### Debug a pipeline log
```bash
curl -X POST http://localhost:8000/debug \
  -H "Content-Type: application/json" \
  -d '{"message": "Check example/sample_logs/spark_error.log and diagnose it"}'
```

### Multi-turn conversation (pass thread_id to continue)
```bash
curl -X POST http://localhost:8000/debug \
  -H "Content-Type: application/json" \
  -d '{"message": "Which config file do I need to edit?", "thread_id": "your-thread-id"}'
```

### Paste raw log content
```bash
curl -X POST http://localhost:8000/debug/log \
  -H "Content-Type: application/json" \
  -d '{"log_content": "ERROR: java.lang.OutOfMemoryError: GC overhead limit exceeded"}'
```

## Environment Variables

GROQ_API_KEY=your_groq_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

## Key Design Decisions

- **LangGraph over raw ReAct loop** — built-in state management, conditional edges, and MemorySaver checkpointing for cross-turn memory
- **temperature=0 for tool calling** — higher temperature causes malformed tool call syntax (tool_use_failed errors)
- **Retry logic on tool_use_failed** — agent_node catches the error and retries once with the LLM without tools
- **FastAPI instance named `api`** — avoids naming conflict with LangGraph's compiled `app` object
- **thread_id per request** — each API session gets its own conversation thread; pass the same thread_id to continue a conversation
- **LangFuse flush as background task** — observability logging doesn't block the API response