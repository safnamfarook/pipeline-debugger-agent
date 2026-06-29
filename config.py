import os
from dotenv import load_dotenv

load_dotenv()

# ── API KEYS ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0

# ── AGENT ─────────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 10  # prevent infinite loops

# ── PATHS ─────────────────────────────────────────────────────────────────────
SAMPLE_LOGS_DIR = "example/sample_logs"