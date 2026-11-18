import os
import sys
from dotenv import load_dotenv

load_dotenv()

from config import LLM_MODEL
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import uuid

# Import agent — note: agent.py calls its LangGraph compiled graph "app"
# We import only the chat function to avoid naming conflicts
from agent import chat
from langfuse import Langfuse

langfuse = Langfuse()

# ── FASTAPI APP ───────────────────────────────────────────────────────────────
api = FastAPI(
    title="Intelligent Data Pipeline Debugger",
    description="AI agent that diagnoses Spark, Airflow, and Kafka pipeline failures",
    version="1.0.0"
)

# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────

class DebugRequest(BaseModel):
    message: str
    thread_id: str = None  # optional — auto-generated if not provided

class DebugResponse(BaseModel):
    response: str
    thread_id: str  # returned so client can continue the conversation

class LogDebugRequest(BaseModel):
    log_content: str
    thread_id: str = None

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str

# ── BACKGROUND TASK ───────────────────────────────────────────────────────────

def flush_langfuse():
    """Flush LangFuse traces after response is sent"""
    langfuse.flush()

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@api.get("/health", response_model=HealthResponse)
async def health():
    """Health check — used by Docker and load balancers"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model=LLM_MODEL
    )


@api.post("/debug", response_model=DebugResponse)
async def debug(request: DebugRequest, background_tasks: BackgroundTasks):
    """
    Chat with the pipeline debugger agent.
    Pass the same thread_id across multiple calls to maintain memory.
    Omit thread_id to start a fresh conversation.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    # Auto-generate thread_id if not provided
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        response = chat(
            user_message=request.message,
            thread_id=thread_id
        )

        # Flush LangFuse after response is sent — non-blocking
        background_tasks.add_task(flush_langfuse)

        return DebugResponse(
            response=response,
            thread_id=thread_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@api.post("/debug/log", response_model=DebugResponse)
async def debug_log(request: LogDebugRequest, background_tasks: BackgroundTasks):
    """
    Convenience endpoint — paste raw log content directly.
    Wraps it into a natural message and sends to the agent.
    """
    if not request.log_content.strip():
        raise HTTPException(status_code=400, detail="log_content cannot be empty")

    thread_id = request.thread_id or str(uuid.uuid4())

    # Wrap raw log in a natural message the agent understands
    message = f"Please diagnose this pipeline log:\n\n{request.log_content}"

    try:
        response = chat(
            user_message=message,
            thread_id=thread_id
        )

        background_tasks.add_task(flush_langfuse)

        return DebugResponse(
            response=response,
            thread_id=thread_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@api.post("/debug/stream")
async def debug_stream(request: DebugRequest):
    """
    Streaming endpoint — for frontends that want token-by-token responses.
    Note: uses the LLM directly without tools for streaming simplicity.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    from langchain_groq import ChatGroq
    from config import LLM_MODEL

    streaming_llm = ChatGroq(model=LLM_MODEL, temperature=0)

    async def token_generator():
        async for chunk in streaming_llm.astream(
            f"You are a data pipeline debugging expert. Analyse this issue:\n{request.message}"
        ):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(token_generator(), media_type="text/plain")


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Pipeline Debugger API...")
    print("Docs: http://127.0.0.1:8000/docs")
    print("\nTest commands (run in a second terminal):")
    print("""
# Health
curl http://127.0.0.1:8000/health

# Debug with a message
curl -X POST http://127.0.0.1:8000/debug \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Check example/sample_logs/spark_error.log and diagnose it"}'

# Debug with raw log content
curl -X POST http://127.0.0.1:8000/debug/log \\
  -H "Content-Type: application/json" \\
  -d '{"log_content": "ERROR: Task failed: java.lang.OutOfMemoryError: GC overhead limit exceeded at org.apache.spark.executor"}'

# Multi-turn — paste the thread_id from first response here
curl -X POST http://127.0.0.1:8000/debug \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Which config file do I need to edit?", "thread_id": "PASTE_THREAD_ID_HERE"}'
""")
    uvicorn.run(api, host="0.0.0.0", port=8000)