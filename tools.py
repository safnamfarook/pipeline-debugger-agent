import os
from langchain_core.tools import tool
from config import SAMPLE_LOGS_DIR

# ── TOOL 1: READ LOG FILE ─────────────────────────────────────────────────────
@tool
def read_log_file(file_path: str) -> str:
    """Read the contents of a log file or DAG file from disk.
    Use this to read Spark error logs, Airflow logs, or any pipeline-related file.
    The file_path should be the full or relative path to the file."""
    try:
        if not os.path.exists(file_path):
            # Try looking in sample logs dir
            fallback = os.path.join(SAMPLE_LOGS_DIR, file_path)
            if os.path.exists(fallback):
                file_path = fallback
            else:
                return f"File not found: {file_path}. Available sample logs are in {SAMPLE_LOGS_DIR}/"
        with open(file_path, "r") as f:
            content = f.read()
        return f"FILE CONTENTS ({file_path}):\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ── TOOL 2: IDENTIFY ERROR TYPE ───────────────────────────────────────────────
@tool
def identify_error_type(error_text: str) -> str:
    """Analyse error text and identify the type of pipeline error.
    Pass the raw error message or log content to get a structured error classification."""
    error_text_lower = error_text.lower()

    # Spark errors
    if any(k in error_text_lower for k in ["outofmemoryerror", "gc overhead", "memory overhead", "container killed by yarn"]):
        return "SPARK_MEMORY_ERROR: Spark executor or driver ran out of memory. Common in large dataset processing."
    if any(k in error_text_lower for k in ["nullpointerexception", "null pointer"]):
        return "SPARK_NULL_ERROR: NullPointerException in Spark — likely a missing null check on a DataFrame column."
    if any(k in error_text_lower for k in ["task failed", "sparkexception", "stage failed"]):
        return "SPARK_TASK_ERROR: Spark task failure — could be data skew, corrupt records, or resource exhaustion."

    # Airflow errors
    if any(k in error_text_lower for k in ["zerodivisionerror", "division by zero"]):
        return "AIRFLOW_DIVISION_ERROR: ZeroDivisionError in an Airflow task — likely missing validation for zero/null denominators."
    if any(k in error_text_lower for k in ["airflow", "taskinstance", "dag"]):
        return "AIRFLOW_TASK_ERROR: Airflow task failure — check the task function logic and upstream dependencies."

    # Kafka errors
    if any(k in error_text_lower for k in ["kafka", "consumer group", "offset", "broker"]):
        return "KAFKA_ERROR: Kafka consumer/producer error — check broker connectivity and consumer group offsets."

    return "UNKNOWN_ERROR: Could not classify the error. Please provide more context or the full log file."

# ── TOOL 3: GET FIX SUGGESTION ────────────────────────────────────────────────
@tool
def get_fix_suggestion(error_type: str) -> str:
    """Get a detailed fix suggestion for a classified error type.
    Pass the error type string returned by identify_error_type."""
    fixes = {
        "SPARK_MEMORY_ERROR": """
FIX — Spark Memory Error:
1. Increase executor memory: set spark.executor.memory=4g in spark-defaults.conf
2. Increase memory overhead: set spark.executor.memoryOverhead=512
3. Reduce partition size: repartition your DataFrame before heavy operations
4. Enable off-heap memory: set spark.memory.offHeap.enabled=true, spark.memory.offHeap.size=2g
5. Check for data skew: use salting technique on skewed join keys
        """,
        "SPARK_NULL_ERROR": """
FIX — Spark NullPointerException:
1. Add null checks before processing: df = df.filter(col("column_name").isNotNull())
2. Use coalesce for default values: df.withColumn("col", coalesce(col("col"), lit("default")))
3. Validate schema at ingestion time before transformations
4. Add try-except in UDFs to handle null values gracefully
        """,
        "SPARK_TASK_ERROR": """
FIX — Spark Task Failure:
1. Check for corrupt records: use spark.read.option("mode", "DROPMALFORMED")
2. Handle data skew: repartition on a more evenly distributed key
3. Increase task timeout: set spark.task.maxFailures=4
4. Check executor logs for the specific failing task
        """,
        "AIRFLOW_DIVISION_ERROR": """
FIX — Airflow ZeroDivisionError:
1. Add validation before division: if record["count"] == 0: return None
2. Use safe division: value / count if count else 0
3. Add data quality checks as upstream tasks in the DAG
4. Log and skip bad records instead of failing the whole task
        """,
        "AIRFLOW_TASK_ERROR": """
FIX — Airflow Task Failure:
1. Check task logs in Airflow UI for the root exception
2. Verify upstream task dependencies completed successfully
3. Add retry logic: retries=3, retry_delay=timedelta(minutes=5) in task definition
4. Check database/API connections used by the task
        """,
        "KAFKA_ERROR": """
FIX — Kafka Error:
1. Check broker connectivity: kafka-broker-api-versions.sh --bootstrap-server host:port
2. Reset consumer group offsets if lag is too high: kafka-consumer-groups.sh --reset-offsets
3. Increase consumer timeout: set session.timeout.ms=30000
4. Check partition assignment and rebalancing logs
        """
    }

    for key in fixes:
        if key in error_type:
            return fixes[key]
    return "No specific fix found. Please share the full error log for detailed analysis."

# ── TOOL 4: LIST SAMPLE LOGS ──────────────────────────────────────────────────
@tool
def list_available_logs() -> str:
    """List all available sample log files the agent can read.
    Use this when the user hasn't specified a file path."""
    try:
        files = os.listdir(SAMPLE_LOGS_DIR)
        if not files:
            return f"No sample logs found in {SAMPLE_LOGS_DIR}"
        file_list = "\n".join([f"- {SAMPLE_LOGS_DIR}/{f}" for f in files])
        return f"Available log files:\n{file_list}"
    except Exception as e:
        return f"Error listing logs: {str(e)}"

# ── EXPORT ALL TOOLS ──────────────────────────────────────────────────────────
all_tools = [read_log_file, identify_error_type, get_fix_suggestion, list_available_logs]