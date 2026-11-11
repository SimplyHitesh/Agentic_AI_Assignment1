"""
finance_agent/config.py
-----------------------
Central configuration loaded from environment variables and session defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project Root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── LLM Configuration ────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")  # Supports local LLMs (LM Studio, Ollama, etc.)
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ── ChromaDB Configuration ───────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(_PROJECT_ROOT / "finance_agent" / "data" / "chroma_db"),
)
CHROMA_COLLECTION_NAME: str = "finance_guidelines"

# ── Supported File Formats ───────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: list[str] = [".csv", ".pdf"]

# ── Standard Spending Categories ─────────────────────────────────────────────
SPENDING_CATEGORIES: list[str] = [
    "Food & Dining",
    "Groceries",
    "Housing & Rent",
    "Utilities & Bills",
    "Transportation",
    "Shopping & Retail",
    "Entertainment & Leisure",
    "Healthcare & Medical",
    "Education & Courses",
    "Travel & Vacation",
    "Subscriptions & Memberships",
    "Financial & Loan EMI",
    "Income & Salary",
    "Transfers",
    "Other",
]

# ── Debug & Tracing ──────────────────────────────────────────────────────────
LANGCHAIN_VERBOSE: bool = os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true"
