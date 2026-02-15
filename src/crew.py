"""
Crew orchestration for the Stat Engine pipeline.
Wires agents, tasks, and tools into a sequential CrewAI workflow.
"""

import os
import re
import time
import yaml
from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM

from src.config.settings import get_settings
from src.tools.csv_reader import csv_reader_tool
from src.tools.data_cleaner import data_cleaner_tool
from src.tools.mysql_loader import mysql_schema_designer_tool
from src.tools.statistical_tests import statistical_test_tool
from src.tools.chart_generator import chart_generator_tool


# Path to YAML configs
CONFIG_DIR = Path(__file__).parent / "config"

# Retry settings for quota errors
MAX_RETRIES = 5
BASE_DELAY = 60  # seconds -- starts at 60s, doubles each retry


def _load_yaml(filename: str) -> dict:
    """Load a YAML config file."""
    filepath = CONFIG_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_llm() -> LLM:
    """Create the Gemini LLM instance for agents."""
    settings = get_settings()
    return LLM(
        model="gemini/gemini-2.0-flash",
        api_key=settings.gemini_api_key,
    )


def build_crew(csv_path: str) -> Crew:
    """
    Build and return a Crew configured for the full analysis pipeline.

    Args:
        csv_path: Absolute path to the uploaded CSV file.
    """
    settings = get_settings()
    llm = _get_llm()

    # Derive per-CSV run folder name
    csv_stem = Path(csv_path).stem
    # Strip UUID prefix if present (from sanitize_filename)
    parts = csv_stem.split("_", 1)
    if len(parts) > 1 and len(parts[0]) >= 8:
        csv_stem = parts[1]
    csv_stem = csv_stem.replace("cleaned_", "")

    # Per-run output paths
    charts_dir = str(settings.run_charts_dir(csv_stem))
    report_path = str(settings.run_report_path(csv_stem))

    # Load agent and task configs
    agent_configs = _load_yaml("agents.yaml")
    task_configs = _load_yaml("tasks.yaml")

    # --- Create Agents ---

    schema_analyst = Agent(
        role=agent_configs["schema_analyst"]["role"],
        goal=agent_configs["schema_analyst"]["goal"],
        backstory=agent_configs["schema_analyst"]["backstory"],
        tools=[csv_reader_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    data_cleaner = Agent(
        role=agent_configs["data_cleaner"]["role"],
        goal=agent_configs["data_cleaner"]["goal"],
        backstory=agent_configs["data_cleaner"]["backstory"],
        tools=[data_cleaner_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    db_architect = Agent(
        role=agent_configs["db_architect"]["role"],
        goal=agent_configs["db_architect"]["goal"],
        backstory=agent_configs["db_architect"]["backstory"],
        tools=[mysql_schema_designer_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    statistical_analyst = Agent(
        role=agent_configs["statistical_analyst"]["role"],
        goal=agent_configs["statistical_analyst"]["goal"],
        backstory=agent_configs["statistical_analyst"]["backstory"],
        tools=[statistical_test_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    report_generator = Agent(
        role=agent_configs["report_generator"]["role"],
        goal=agent_configs["report_generator"]["goal"],
        backstory=agent_configs["report_generator"]["backstory"],
        tools=[chart_generator_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # --- Create Tasks ---
    # Template variables are replaced in task descriptions

    schema_task = Task(
        description=task_configs["schema_analysis"]["description"].format(csv_path=csv_path),
        expected_output=task_configs["schema_analysis"]["expected_output"],
        agent=schema_analyst,
    )

    cleaning_task = Task(
        description=task_configs["data_cleaning"]["description"].format(csv_path=csv_path),
        expected_output=task_configs["data_cleaning"]["expected_output"],
        agent=data_cleaner,
        context=[schema_task],
    )

    db_task = Task(
        description=task_configs["database_loading"]["description"].format(cleaned_csv_path=csv_path),
        expected_output=task_configs["database_loading"]["expected_output"],
        agent=db_architect,
        context=[cleaning_task],
    )

    stats_task = Task(
        description=task_configs["statistical_analysis"]["description"].format(cleaned_csv_path=csv_path),
        expected_output=task_configs["statistical_analysis"]["expected_output"],
        agent=statistical_analyst,
        context=[cleaning_task],
    )

    report_task = Task(
        description=task_configs["report_generation"]["description"],
        expected_output=task_configs["report_generation"]["expected_output"],
        agent=report_generator,
        context=[schema_task, cleaning_task, db_task, stats_task],
        output_file=report_path,
    )

    # --- Assemble Crew ---

    crew = Crew(
        agents=[schema_analyst, data_cleaner, db_architect, statistical_analyst, report_generator],
        tasks=[schema_task, cleaning_task, db_task, stats_task, report_task],
        process=Process.sequential,
        verbose=True,
        max_rpm=5,  # Conservative -- safe for Gemini free tier new accounts
    )

    return crew


def _is_quota_error(error: Exception) -> bool:
    """Check if an error is a rate limit / quota error."""
    # Check the full exception chain (CrewAI wraps errors)
    errors_to_check = [error]
    if error.__cause__:
        errors_to_check.append(error.__cause__)
    if error.__context__:
        errors_to_check.append(error.__context__)

    keywords = [
        "quota", "rate limit", "429", "resource exhausted", "too many requests",
        "retrydelay", "retryinfo", "exhausted", "resource_exhausted",
    ]
    for err in errors_to_check:
        error_str = str(err).lower()
        if any(keyword in error_str for keyword in keywords):
            return True
    return False


def _extract_retry_delay(error: Exception) -> int | None:
    """Try to extract the suggested retry delay from the error message."""
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)s", str(error))
    if match:
        return int(match.group(1))
    return None


def run_analysis(csv_path: str, status_callback=None) -> dict:
    """
    Run the full analysis pipeline with automatic retry on quota errors.

    Args:
        csv_path: Absolute path to the CSV file.
        status_callback: Optional callable for status updates (e.g., Streamlit st.write).

    Returns:
        Dict with keys: report (str), run_dir (str), charts_dir (str).
    """
    settings = get_settings()

    # Derive per-CSV run folder name (same logic as build_crew)
    csv_stem = Path(csv_path).stem
    parts = csv_stem.split("_", 1)
    if len(parts) > 1 and len(parts[0]) >= 8:
        csv_stem = parts[1]
    csv_stem = csv_stem.replace("cleaned_", "")

    run_dir = str(settings.run_dir(csv_stem))
    charts_dir = str(settings.run_charts_dir(csv_stem))

    crew = build_crew(csv_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = crew.kickoff()
            return {
                "report": str(result),
                "run_dir": run_dir,
                "charts_dir": charts_dir,
            }
        except Exception as e:
            if _is_quota_error(e) and attempt < MAX_RETRIES:
                # Use API-suggested delay or exponential backoff
                suggested = _extract_retry_delay(e)
                delay = suggested if suggested else BASE_DELAY * (2 ** (attempt - 1))
                msg = f"[Retry {attempt}/{MAX_RETRIES}] Quota limit hit. Waiting {delay}s before retrying..."
                print(msg)
                if status_callback:
                    status_callback(msg)
                time.sleep(delay)
            else:
                raise

