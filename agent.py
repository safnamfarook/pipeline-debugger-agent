import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from config import LLM_MODEL, LLM_TEMPERATURE, MAX_ITERATIONS
from tools import all_tools

load_dotenv()

# ── LANGFUSE ──────────────────────────────────────────────────────────────────
langfuse = Langfuse()
langfuse_handler = CallbackHandler()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
llm_with_tools = llm.bind_tools(all_tools)
tool_map = {t.name: t for t in all_tools}

# ── STATE ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: int

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert data pipeline debugging assistant with deep knowledge of:
- Apache Spark (memory management, task failures, data skew)
- Apache Airflow (DAG failures, task errors, dependencies)
- Apache Kafka (consumer/producer issues, offset management)

When a user reports a pipeline issue:
1. If they mention a file, use read_log_file to read it first
2. If no file is mentioned, use list_available_logs to show what's available
3. Use identify_error_type to classify the error from the log content
4. Use get_fix_suggestion to get detailed fix steps for the error type
5. Summarise the findings clearly with the error cause and fix steps

Always be specific — reference the actual error messages from the logs in your answer.
"""

# ── NODES ─────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    """LLM reasons and decides what to do next"""
    system = SystemMessage(content=SYSTEM_PROMPT)
    try:
        response = llm_with_tools.invoke(
            [system] + state["messages"],
            config={"callbacks": [langfuse_handler]}
        )
    except Exception as e:
        error_text = str(e)
        # Groq reports a malformed tool call in two ways, depending on the model:
        #   "tool_use_failed"     (seen with llama-3.3-70b-versatile)
        #   "output_parse_failed" (seen with openai/gpt-oss-120b)
        is_malformed_tool_call = (
            "tool_use_failed" in error_text
            or "output_parse_failed" in error_text
        )
        if is_malformed_tool_call:
            try:
                # First retry: same call WITH tools, so the agent can still
                # reach get_fix_suggestion
                response = llm_with_tools.invoke(
                    [system] + state["messages"],
                    config={"callbacks": [langfuse_handler]}
                )
            except Exception:
                try:
                    # Second retry: WITHOUT tools, so the user still gets an answer
                    response = llm.invoke(
                        [system] + state["messages"],
                        config={"callbacks": [langfuse_handler]}
                    )
                except Exception:
                    # Last resort: a clear message instead of crashing
                    response = AIMessage(
                        content="Sorry, I couldn't finish the analysis because the model "
                                "returned a malformed tool call. Please try again."
                    )
        else:
            raise e
    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1
    }

def tool_node(state: AgentState) -> AgentState:
    """Execute whatever tool the LLM called"""
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = tool_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        print(f"\n  → TOOL: {tool_call['name']}({tool_call['args']})")
        print(f"  → RESULT: {str(result)[:200]}...")
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    return {"messages": results}

# ── CONDITIONAL EDGE ──────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Continue if LLM made tool calls and under iteration limit, else end"""
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        print(f"\n⚠️  Max iterations ({MAX_ITERATIONS}) reached")
        return END
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# ── GRAPH ─────────────────────────────────────────────────────────────────────
checkpointer = MemorySaver()

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=checkpointer)

# ── CHAT INTERFACE ────────────────────────────────────────────────────────────
def chat(user_message: str, thread_id: str = "default") -> str:
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print('='*60)

    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "iterations": 0
        },
        config=config
    )

    final = result["messages"][-1].content
    langfuse.flush()
    print(f"\nAGENT:\n{final}")
    return final

# ── TEST RUN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: Spark log
    chat(
        "I have a Spark error log at example/sample_logs/spark_error.log — can you diagnose it?",
        thread_id="test_spark"
    )

    # Test 2: Follow-up (memory test)
    chat(
        "Which config file do I need to edit to fix this?",
        thread_id="test_spark"
    )

    # Test 3: Airflow log
    chat(
        "Now check example/sample_logs/airflow_error.log and tell me what went wrong",
        thread_id="test_airflow"
    )