"""
Unit tests for ParserAgent.
Tests:
- Reading sample CSV
- Detecting date/description/debit/credit headers
- Merging debit/credit into amount
- Coercing dates and sorting ascending
"""

import os
from pathlib import Path
import pytest
import pandas as pd

from finance_agent.agents.parser_agent import ParserAgent, ParseResult

SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "sample_statement.csv"


def test_parser_csv_success():
    agent = ParserAgent()
    assert SAMPLE_CSV_PATH.exists(), f"Sample statement not found at {SAMPLE_CSV_PATH}"

    result = agent.parse(str(SAMPLE_CSV_PATH))
    assert result.success is True
    assert result.df is not None
    assert result.row_count > 0
    assert result.source_format == "csv"

    df = result.df
    # Check expected canonical columns
    for col in ["date", "description", "amount", "transaction_type", "raw_text"]:
        assert col in df.columns, f"Missing column {col}"

    # Dates should be sorted chronologically
    assert df["date"].is_monotonic_increasing

    # Transaction types should match amount signs
    assert (df.loc[df["transaction_type"] == "debit", "amount"] < 0).all()
    assert (df.loc[df["transaction_type"] == "credit", "amount"] >= 0).all()


def test_parser_unsupported_format():
    agent = ParserAgent()
    result = agent.parse("dummy_statement.docx")
    assert result.success is False
    assert "Unsupported file type" in result.error

