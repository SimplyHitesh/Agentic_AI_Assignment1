"""
finance_agent/agents/analysis_agent.py
--------------------------------------
Analysis Agent: Answers user inquiries regarding totals, trends, and breakdowns.
Uses:
  - PandasFinanceTool for deterministic data calculations
  - FinanceCalculator for arithmetic and financial metrics
  - LangChain LCEL pipeline with Chain-of-Thought (CoT) prompting
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from finance_agent.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    LLM_TEMPERATURE,
)
from finance_agent.tools.calculator import FinanceCalculator
from finance_agent.tools.pandas_tool import PandasFinanceTool
from finance_agent.prompts.prompts import ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """
    Executes quantitative analysis on the categorised statement.
    Combines Pandas aggregations, safe calculator computations, and LCEL LLM reasoning.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.llm = self._init_llm()

    def _init_llm(self) -> Optional[ChatOpenAI]:
        """Initializes LangChain ChatOpenAI if credentials exist."""
        if not self.api_key:
            return None
        try:
            kwargs: Dict[str, Any] = {
                "model": OPENAI_MODEL,
                "temperature": LLM_TEMPERATURE,
                "api_key": self.api_key,
            }
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            return ChatOpenAI(**kwargs)
        except Exception as e:
            logger.warning("Could not initialize LLM for AnalysisAgent: %s", e)
            return None

    def analyze(self, df: pd.DataFrame, query: str) -> str:
        """
        Main entry point for spending analysis.
        Calculates exact metrics using PandasFinanceTool, then passes them
        into the LCEL chain (or deterministic fallback) for formatted output.
        """
        if df.empty:
            return "No transactions are available to analyze. Please upload a statement first."

        # 1. Compute exact metrics with Pandas
        kpis = PandasFinanceTool.compute_kpis(df)
        cat_df = PandasFinanceTool.get_category_breakdown(df)
        monthly_df = PandasFinanceTool.get_monthly_breakdown(df)
        top5_df = PandasFinanceTool.get_top_transactions(df, n=5)

        # 2. Build structured text summary
        summary_text = self._build_data_context(kpis, cat_df, monthly_df, top5_df)

        # 3. If LLM is available, invoke LCEL chain
        if self.llm:
            try:
                chain = ANALYSIS_PROMPT | self.llm | StrOutputParser()
                response = chain.invoke({
                    "statement_summary": summary_text,
                    "user_message": query,
                })
                return response
            except Exception as exc:
                logger.warning("LLM Analysis failed (%s); reverting to deterministic analysis.", exc)

        # 4. Deterministic fallback if LLM offline / no key
        return self._generate_deterministic_analysis(df, query, kpis, cat_df, top5_df)

    def _build_data_context(
        self,
        kpis: Dict[str, Any],
        cat_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
        top5_df: pd.DataFrame,
    ) -> str:
        """Constructs an exact textual summary of the dataset for the LLM."""
        lines = [
            f"- Statement Period: {kpis['date_min']} to {kpis['date_max']}",
            f"- Total Transactions: {kpis['transaction_count']}",
            f"- Total Income (Credits): ₹{kpis['total_income']:,.2f}",
            f"- Total Expenses (Debits): ₹{kpis['total_expense']:,.2f}",
            f"- Net Balance / Savings: ₹{kpis['net_savings']:,.2f}",
            f"- Overall Savings Rate: {kpis['savings_rate_pct']}%",
            f"- Average Debit Size: ₹{kpis['avg_debit']:,.2f}",
            "\n### Category Breakdown (Sorted by Spending):",
        ]

        if not cat_df.empty:
            for _, row in cat_df.iterrows():
                lines.append(
                    f"  * {row['category']}: ₹{row['spent']:,.2f} ({row['percentage']}% of total expenses, {row['count']} txns)"
                )
        else:
            lines.append("  (No categorized expenses recorded)")

        if not top5_df.empty:
            lines.append("\n### Top 5 Largest Debits:")
            for _, row in top5_df.iterrows():
                dt_str = row["date"].strftime("%d %b %Y") if pd.notnull(row["date"]) else "N/A"
                lines.append(
                    f"  * {dt_str} | ₹{abs(row['amount']):,.2f} | {row['description']} ({row.get('category', 'Other')})"
                )

        if not monthly_df.empty:
            lines.append("\n### Monthly Aggregates:")
            for _, row in monthly_df.iterrows():
                lines.append(
                    f"  * {row['month']}: Inflow ₹{row['income']:,.2f} | Outflow ₹{row['expense']:,.2f} | Net ₹{row['net']:,.2f}"
                )

        return "\n".join(lines)

    def _generate_deterministic_analysis(
        self,
        df: pd.DataFrame,
        query: str,
        kpis: Dict[str, Any],
        cat_df: pd.DataFrame,
        top5_df: pd.DataFrame,
    ) -> str:
        """Deterministic response generator when running without LLM API key."""
        q_lower = query.lower()

        # Check for specific category query
        matching_cat = None
        for cat in cat_df["category"].tolist() if not cat_df.empty else []:
            if cat.lower() in q_lower:
                matching_cat = cat
                break

        if matching_cat:
            cat_row = cat_df[cat_df["category"] == matching_cat].iloc[0]
            cat_txns = df[(df["category"] == matching_cat) & (df["transaction_type"] == "debit")]
            lines = [
                f"### 📊 Spending Analysis for **{matching_cat}**",
                f"- **Total Spent:** ₹{cat_row['spent']:,.2f}",
                f"- **Percentage of Total Expenses:** {cat_row['percentage']}%",
                f"- **Number of Transactions:** {cat_row['count']}",
                "\n**Transactions in this category:**",
            ]
            for _, row in cat_txns.iterrows():
                dt = row["date"].strftime("%d %b %Y") if pd.notnull(row["date"]) else "N/A"
                lines.append(f"- {dt} | **₹{abs(row['amount']):,.2f}** | {row['description']}")
            return "\n".join(lines)

        # General Spending Summary
        lines = [
            "### 📈 Statement Financial Summary",
            f"- **Period:** {kpis['date_min']} → {kpis['date_max']}",
            f"- **Total Income:** ₹{kpis['total_income']:,.2f}",
            f"- **Total Expenses:** ₹{kpis['total_expense']:,.2f}",
            f"- **Net Balance:** ₹{kpis['net_savings']:,.2f}",
            f"- **Savings Rate:** {kpis['savings_rate_pct']}%",
            "\n#### Top Spending Categories:",
        ]
        if not cat_df.empty:
            for _, row in cat_df.head(5).iterrows():
                lines.append(f"- **{row['category']}:** ₹{row['spent']:,.2f} ({row['percentage']}%)")

        if not top5_df.empty:
            lines.append("\n#### Top 5 Largest Expenses:")
            for _, row in top5_df.iterrows():
                dt = row["date"].strftime("%d %b %Y") if pd.notnull(row["date"]) else "N/A"
                lines.append(f"- {dt} | **₹{abs(row['amount']):,.2f}** | {row['description']}")

        return "\n".join(lines)
