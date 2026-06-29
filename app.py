import os
import tempfile
import streamlit as st
from langchain_core.messages import HumanMessage
from agent import app, langfuse

st.set_page_config(
    page_title="Data Pipeline Debugger",
    layout="wide",
    page_icon="🔧"
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_1"
if "active_log_path" not in st.session_state:
    st.session_state.active_log_path = None
if "active_log_name" not in st.session_state:
    st.session_state.active_log_name = None

# ── HELPER ────────────────────────────────────────────────────────────────────
def run_agent(user_message: str) -> str:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "iterations": 0
        },
        config=config
    )
    langfuse.flush()
    return result["messages"][-1].content

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔧 Intelligent Data Pipeline Debugger")
st.caption("Upload or paste a Spark, Airflow, or Kafka log — then ask questions to diagnose and fix issues")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Load a Log File")

    tab1, tab2 = st.tabs(["📄 Upload File", "📋 Paste Log"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a .log or .txt file",
            type=["log", "txt"]
        )
        if st.button("Load Uploaded File", type="primary"):
            if uploaded_file:
                # Save to system temp dir — not inside project
                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".log"
                )
                tmp.write(uploaded_file.read())
                tmp.close()
                st.session_state.active_log_path = tmp.name
                st.session_state.active_log_name = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.thread_id = f"session_{uploaded_file.name}"
                st.success(f"✅ Loaded: {uploaded_file.name}")
            else:
                st.warning("Please upload a file first")

    with tab2:
        pasted_content = st.text_area(
            "Paste your error log here",
            height=200,
            placeholder="Paste Spark, Airflow, or Kafka error logs..."
        )
        if st.button("Load Pasted Log", type="primary"):
            if pasted_content.strip():
                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".log"
                )
                tmp.write(pasted_content.encode())
                tmp.close()
                st.session_state.active_log_path = tmp.name
                st.session_state.active_log_name = "pasted_log.log"
                st.session_state.chat_history = []
                st.session_state.thread_id = f"session_pasted_{len(pasted_content)}"
                st.success("✅ Pasted log loaded")
            else:
                st.warning("Please paste some log content first")

    st.divider()

    # Show active log
    if st.session_state.active_log_path:
        st.markdown("**Active log:**")
        st.markdown(f"- 📄 {st.session_state.active_log_name}")

    # Clear button
    if st.button("🗑️ Clear Session"):
        st.session_state.chat_history = []
        st.session_state.thread_id = "session_1"
        st.session_state.active_log_path = None
        st.session_state.active_log_name = None
        st.rerun()

# ── MAIN CHAT AREA ────────────────────────────────────────────────────────────
if not st.session_state.active_log_path:
    st.info("👈 Upload a log file or paste log content in the sidebar to get started")
else:
    # Auto-suggest first message
    if not st.session_state.chat_history:
        st.info(f"💡 Try asking: *'What went wrong and how do I fix it?'* or *'Diagnose this error'*")

    # Chat input
    query = st.chat_input("Ask about your pipeline error...")

    if query:
        # Inject log path on first message
        if not st.session_state.chat_history:
            full_query = f"{query}\n\n(Log file is at: {st.session_state.active_log_path})"
        else:
            full_query = query

        with st.spinner("Debugging..."):
            response = run_agent(full_query)

        st.session_state.chat_history.append((query, response))

    # Display chat history
    for user_msg, agent_msg in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(agent_msg)