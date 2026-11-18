"""
run_agent_eval.py

Evaluation for the Intelligent Data Pipeline Debugger.

What this script does, for each of the 20 failure scenarios:
  1. RULES ONLY: passes the whole log straight into identify_error_type()
     with no LLM involved. This shows how good the keyword rules are alone.
  2. FULL AGENT: asks your real LangGraph agent to diagnose the log file,
     in a brand new conversation thread each time.
  3. Records which tools the agent called, what error type its tool call
     returned, what text it passed to the tool, and its final answer.

Everything is saved to eval_results/agent_results.csv.
The final answers are then marked by reading them (the next step).

How to run it (from your project folder, with your venv active):
    python run_agent_eval.py
"""

import os
import csv
import time
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Your real agent and your real tool, unchanged
from agent import app, langfuse
from tools import identify_error_type
from config import LLM_MODEL, MAX_ITERATIONS


# ── SETTINGS ──────────────────────────────────────────────────────────────────

SCENARIOS_CSV = os.path.join("eval_data", "scenarios.csv")
SCENARIOS_FOLDER = os.path.join("eval_data", "scenarios")
RESULTS_PATH = os.path.join("eval_results", "agent_results.csv")

# Each scenario makes several Groq calls (one per agent step), so we wait
# between scenarios to stay under the free tier limits.
SECONDS_BETWEEN_SCENARIOS = 20

# If Groq says we hit the rate limit, wait this long and try once more
SECONDS_TO_WAIT_ON_RATE_LIMIT = 60


# ── HELPER: TURN A TOOL RESULT INTO A SHORT LABEL ─────────────────────────────
# identify_error_type returns text like
#   "SPARK_MEMORY_ERROR: Spark executor or driver ran out of memory..."
# We only want the part before the first colon: "SPARK_MEMORY_ERROR".

def get_label(tool_result_text):
    label = tool_result_text.split(":")[0]
    label = label.strip()
    return label


# ── STEP 1: RULES ONLY ────────────────────────────────────────────────────────

def run_rules_only(log_text):
    result_text = identify_error_type.invoke({"error_text": log_text})
    return get_label(result_text)


# ── STEP 2: RUN THE FULL AGENT ON ONE LOG ─────────────────────────────────────

def run_agent_once(log_path, thread_id):
    user_message = (
        f"My pipeline failed. Please diagnose the log at {log_path} "
        f"and tell me the cause and how to fix it."
    )

    # A fresh thread_id means a fresh memory, so earlier scenarios
    # cannot influence this one.
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "iterations": 0,
        },
        config=config,
    )
    return result


def run_agent_with_retry(log_path, thread_id):
    try:
        return run_agent_once(log_path, thread_id)
    except Exception as first_error:
        error_text = str(first_error).lower()
        if "rate_limit" in error_text or "429" in error_text:
            print(f"   Rate limit hit. Waiting {SECONDS_TO_WAIT_ON_RATE_LIMIT} seconds and trying once more...")
            time.sleep(SECONDS_TO_WAIT_ON_RATE_LIMIT)
            # Use a new thread id so the retry starts clean
            return run_agent_once(log_path, thread_id + "_retry")
        raise first_error


# ── STEP 3: READ WHAT HAPPENED INSIDE THE AGENT RUN ───────────────────────────

def summarise_agent_run(result):
    messages = result["messages"]

    # First pass: remember the name of every tool call by its id
    tool_name_by_id = {}
    tools_called_in_order = []
    identify_inputs = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_name_by_id[tool_call["id"]] = tool_call["name"]
                tools_called_in_order.append(tool_call["name"])

                # Save what text the agent passed to identify_error_type,
                # so we can see if it sent the whole log or just one line
                if tool_call["name"] == "identify_error_type":
                    passed_text = str(tool_call["args"].get("error_text", ""))
                    identify_inputs.append(passed_text)

    # Second pass: find the results that came back from identify_error_type
    identify_labels = []
    for message in messages:
        if isinstance(message, ToolMessage):
            tool_name = tool_name_by_id.get(message.tool_call_id, "")
            if tool_name == "identify_error_type":
                identify_labels.append(get_label(message.content))

    final_answer = messages[-1].content
    iterations = result.get("iterations", 0)

    return {
        "tools_called": " > ".join(tools_called_in_order),
        "identify_labels": "; ".join(identify_labels),
        "last_identify_label": identify_labels[-1] if identify_labels else "NOT_CALLED",
        "identify_input": " ||| ".join(identify_inputs),
        "final_answer": final_answer,
        "iterations": iterations,
        "hit_max_iterations": iterations >= MAX_ITERATIONS,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    scenarios = []
    with open(SCENARIOS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append(row)
    print(f"Loaded {len(scenarios)} scenarios\n")

    run_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results = []

    for number, scenario in enumerate(scenarios, start=1):
        log_path = os.path.join(SCENARIOS_FOLDER, scenario["log_file"])
        print(f"[{number}/{len(scenarios)}] {scenario['id']}: {scenario['log_file']}")

        with open(log_path, "r") as f:
            log_text = f.read()

        # 1. Rules only (no LLM)
        rules_label = run_rules_only(log_text)

        # 2. Full agent
        row = {
            "id": scenario["id"],
            "group": scenario["group"],
            "expected_error_type": scenario["expected_error_type"],
            "expected_root_cause": scenario["expected_root_cause"],
            "rules_only_label": rules_label,
            "rules_only_correct": rules_label == scenario["expected_error_type"],
        }

        thread_id = f"eval_{run_stamp}_{scenario['id']}"
        try:
            result = run_agent_with_retry(log_path, thread_id)
            summary = summarise_agent_run(result)
            row.update(summary)
            row["agent_tool_label_correct"] = summary["last_identify_label"] == scenario["expected_error_type"]
            row["error"] = ""
        except Exception as e:
            # Record the crash and carry on with the next scenario
            print(f"   !! Agent crashed: {e}")
            row.update({
                "tools_called": "",
                "identify_labels": "",
                "last_identify_label": "CRASHED",
                "identify_input": "",
                "final_answer": "",
                "iterations": 0,
                "hit_max_iterations": False,
                "agent_tool_label_correct": False,
                "error": str(e),
            })

        print(f"   rules only: {row['rules_only_label']}   agent tool label: {row['last_identify_label']}   expected: {row['expected_error_type']}")
        results.append(row)

        if number < len(scenarios):
            time.sleep(SECONDS_BETWEEN_SCENARIOS)

    langfuse.flush()

    # ── SAVE ──────────────────────────────────────────────────────────────────
    os.makedirs("eval_results", exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nSaved results to {RESULTS_PATH}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    total = len(results)
    rules_correct = 0
    agent_label_correct = 0
    used_all_three_core_tools = 0
    crashes = 0
    hit_max = 0

    for row in results:
        if row["rules_only_correct"]:
            rules_correct = rules_correct + 1
        if row["agent_tool_label_correct"]:
            agent_label_correct = agent_label_correct + 1

        tools = row["tools_called"]
        if "read_log_file" in tools and "identify_error_type" in tools and "get_fix_suggestion" in tools:
            used_all_three_core_tools = used_all_three_core_tools + 1

        if row["error"]:
            crashes = crashes + 1
        if row["hit_max_iterations"]:
            hit_max = hit_max + 1

    print("\n── AGENT EVAL SUMMARY ──")
    print(f"Run date:                         {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"LLM model:                        {LLM_MODEL}")
    print(f"Scenarios:                        {total}")
    print(f"Rules only, correct error type:   {rules_correct} of {total}")
    print(f"Agent's tool label correct:       {agent_label_correct} of {total}")
    print(f"Used read > identify > fix tools: {used_all_three_core_tools} of {total}")
    print(f"Crashes:                          {crashes}")
    print(f"Hit the max iterations limit:     {hit_max}")
    print("Final answers are not marked yet. That is the next step.")


if __name__ == "__main__":
    main()
