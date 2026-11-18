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
Groq (openai/gpt-oss-120b)
↓
LangFuse (observability + tracing)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent framework | LangGraph 1.x |
| LLM | Groq — openai/gpt-oss-120b |
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
├── run_agent_eval.py  # Evaluation on 20 failure scenarios
├── eval_data/         # 20 scenario logs + expected answers (scenarios.csv)
├── eval_results/      # Raw results and marked results
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

## Evaluation

I tested the agent on 20 failure logs, each with a known correct root cause (`eval_data/scenarios.csv`).

| Group | Scenarios | What they cover |
|---|---|---|
| Spark | 7 | executor, YARN and driver memory, null errors, a corrupt Parquet file, data skew |
| Airflow | 5 | divide by zero (2), upstream failure, database connection timeout, missing field |
| Kafka | 4 | rebalance during commit, broker down, offset out of range, message too large |
| Tricky | 2 | a Kafka failure inside an Airflow task, a Spark memory failure launched from Airflow |
| Out of scope | 2 | an S3 permission error and a dbt SQL error, which fit none of the 6 error types |

Each scenario runs in a fresh conversation thread through the real LangGraph agent. Final answers were then marked by reading each one against the expected root cause.

**Results** (run on 18 Nov 2026, model `openai/gpt-oss-120b`)

| Measure | Result |
|---|---|
| Correct root cause in the final answer | **20 of 20** |
| Keyword rules alone (`identify_error_type` on the whole log, no LLM) | 15 of 20 |
| Both out of scope errors named correctly, with no Spark, Airflow or Kafka fix forced | 2 of 2 |
| Crashes | 0 of 20 |

**What I learned**
- **The agent beats its own rules.** Real Airflow logs contain "Task failed with exception", which matches a Spark rule first, so the keyword tool mislabels most Airflow logs. Sometimes the agent called the tool again with just the key error line (for example `KeyError: 'email_address'`) and got the right type. In other cases (A01, A02, A04) it kept a wrong or UNKNOWN label but still diagnosed the right root cause by reading the log itself.
- **The evaluation found a real bug.** After switching models, Groq started reporting malformed tool calls as `output_parse_failed` instead of `tool_use_failed`, so the old retry never ran and the agent crashed. Earlier runs also showed that the last fallback (the LLM without tools) can fail, because `gpt-oss-120b` still tries to call a tool. The agent now catches both error codes, retries with tools, then without tools, and as a last resort returns a clear "couldn't finish the analysis" message instead of crashing. That last resort was not needed in the final run.
- **The diagnosis is more reliable than the fix code.** One answer had a correct diagnosis but a wrong code example (`small_df.broadcast()` is not valid PySpark). Fix steps should be checked before use.
- **Results vary between runs.** The same agent can take different tool paths on different runs. Earlier runs on the previous version of the code scored 19 of 20, each with one crash, so a single run of 20 is a snapshot, not a guarantee.

To rerun:

```bash
python run_agent_eval.py
```

Results are saved to `eval_results/agent_results.csv`. The marked results, with a reason for each mark, are in `eval_results/agent_results_marked.csv`.

> Note: Groq retired `llama-3.3-70b-versatile` on 16 Aug 2026, so the project now uses `openai/gpt-oss-120b`.

## Key Design Decisions

- **LangGraph over raw ReAct loop** — built-in state management, conditional edges, and MemorySaver checkpointing for cross-turn memory
- **temperature=0 for tool calling** — higher temperature causes malformed tool call syntax (tool_use_failed errors)
- **Retry logic on malformed tool calls** — agent_node catches both Groq error codes (`tool_use_failed` and `output_parse_failed`), retries once with tools, then once without tools, and returns a clear error message if both fail
- **FastAPI instance named `api`** — avoids naming conflict with LangGraph's compiled `app` object
- **thread_id per request** — each API session gets its own conversation thread; pass the same thread_id to continue a conversation
- **LangFuse flush as background task** — observability logging doesn't block the API response