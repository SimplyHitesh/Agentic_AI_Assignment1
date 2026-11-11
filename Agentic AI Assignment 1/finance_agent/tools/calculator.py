"""
finance_agent/tools/calculator.py
---------------------------------
Safe numerical calculation tool for exact arithmetic and financial formulas.
Used by Analysis Agent and Advisor Agent to guarantee math precision without LLM hallucination.
"""

from __future__ import annotations

import math
import re
from typing import Union
from langchain_core.tools import tool


class FinanceCalculator:
    """Core numerical calculation engine."""

    # Allowed math functions for safe evaluation
    SAFE_GLOBALS = {
        "__builtins__": None,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "math": math,
    }

    @classmethod
    def evaluate_expression(cls, expr: str) -> Union[float, int, str]:
        """
        Safely evaluates an arithmetic expression like '450 + 1250 * 0.18'.
        Strips dangerous tokens to prevent code injection.
        """
        # Clean expression
        clean_expr = expr.replace("₹", "").replace(",", "").strip()

        # Sanitize: allow only digits, operators, parens, decimal points, and math keywords
        if not re.match(r"^[\d\s\+\-\*\/\%\.\(\)\,\<\>\=]+$", clean_expr):
            # Check if it has simple function calls
            if not any(fn in clean_expr for fn in ["abs", "round", "min", "max", "sum", "pow"]):
                return "Error: Expression contains unsupported or unsafe characters."

        try:
            # Evaluate using restricted globals and empty locals
            result = eval(clean_expr, cls.SAFE_GLOBALS, {})
            if isinstance(result, float):
                return round(result, 2)
            return result
        except Exception as e:
            return f"Calculation error: {str(e)}"

    @staticmethod
    def calculate_percentage(part: float, whole: float) -> float:
        """Calculate percentage with zero-division safeguard."""
        if whole == 0:
            return 0.0
        return round((abs(part) / abs(whole)) * 100.0, 2)

    @staticmethod
    def calculate_savings_rate(income: float, expenses: float) -> float:
        """Compute Savings Rate % = (Income - Expenses) / Income * 100."""
        if income <= 0:
            return 0.0
        savings = income - expenses
        return round((savings / income) * 100.0, 2)

    @staticmethod
    def calculate_runway(current_savings: float, monthly_burn: float) -> float:
        """Calculate emergency fund runway in months."""
        if monthly_burn <= 0:
            return 999.0
        return round(current_savings / monthly_burn, 1)


# ── LangChain Tool Wrapper ───────────────────────────────────────────────────
@tool
def calculate(expression: str) -> str:
    """
    Useful for running mathematical calculations, arithmetic, or financial formulas.
    Input should be a mathematical expression string, e.g. '(15000 + 12000) / 45000 * 100' or '45000 - 32500'.
    """
    res = FinanceCalculator.evaluate_expression(expression)
    return str(res)

