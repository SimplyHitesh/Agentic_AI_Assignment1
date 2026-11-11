"""
Unit tests for FinanceCalculator and PandasFinanceTool.
"""

import pytest
import pandas as pd
from finance_agent.tools.calculator import FinanceCalculator, calculate
from finance_agent.tools.pandas_tool import PandasFinanceTool


def test_calculator_basic_eval():
    res = FinanceCalculator.evaluate_expression("15000 + 12000")
    assert res == 27000

    pct = FinanceCalculator.evaluate_expression("(45000 - 32500) / 45000 * 100")
    assert round(pct, 2) == 27.78


def test_calculator_percentage_and_savings_rate():
    pct = FinanceCalculator.calculate_percentage(5000, 20000)
    assert pct == 25.0

    savings_rate = FinanceCalculator.calculate_savings_rate(50000, 35000)
    assert savings_rate == 30.0

    # Zero income guard
    assert FinanceCalculator.calculate_savings_rate(0, 5000) == 0.0


def test_pandas_finance_tool():
    # Sample DataFrame
    data = {
        "date": pd.to_datetime(["2024-10-01", "2024-10-02", "2024-10-03"]),
        "description": ["Salary", "Zomato", "Rent"],
        "amount": [50000.0, -500.0, -15000.0],
        "transaction_type": ["credit", "debit", "debit"],
        "category": ["Income & Salary", "Food & Dining", "Housing & Rent"],
        "raw_text": ["Salary", "Zomato", "Rent"],
    }
    df = pd.DataFrame(data)

    kpis = PandasFinanceTool.compute_kpis(df)
    assert kpis["total_income"] == 50000.0
    assert kpis["total_expense"] == 15500.0
    assert kpis["net_savings"] == 34500.0
    assert kpis["savings_rate_pct"] == 69.0
    assert kpis["transaction_count"] == 3

    breakdown = PandasFinanceTool.get_category_breakdown(df)
    assert len(breakdown) == 2
    assert "Housing & Rent" in breakdown["category"].values

    top = PandasFinanceTool.get_top_transactions(df, n=1)
    assert len(top) == 1
    assert top.iloc[0]["description"] == "Rent"

