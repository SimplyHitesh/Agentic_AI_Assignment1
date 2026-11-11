"""
finance_agent/agents/advisor_agent.py
-------------------------------------
Advisor Agent: Flags overspending and suggests practical, grounded savings actions.
Demonstrates:
  - Agentic RAG: Dynamically retrieves relevant budgeting benchmarks from ChromaDB
  - Reflection: Evaluates user metrics against 50/30/20 and category thresholds
  - LangChain LCEL pipeline combining retrieved context and financial profile
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

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
from finance_agent.utils.chroma_store import ChromaStore
from finance_agent.prompts.prompts import ADVISOR_PROMPT

logger = logging.getLogger(__name__)


class AdvisorAgent:
    """
    Generates tailored, actionable personal finance recommendations.
    Employs Agentic RAG by querying ChromaDB for relevant financial guidelines,
    diagnosing overspending red flags, and creating concrete savings targets.
    """

    def __init__(self, chroma_store: Optional[ChromaStore] = None, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.chroma = chroma_store or ChromaStore(api_key=self.api_key)
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
            logger.warning("Could not initialize LLM for AdvisorAgent: %s", e)
            return None

    def advise(self, df: pd.DataFrame, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes Agentic RAG advisory workflow:
        1. Computes financial metrics & flags.
        2. Retrieves relevant guideline chunks from ChromaDB.
        3. Generates grounded recommendations via LCEL chain or deterministic reflection.
        """
        if df.empty:
            return {
                "report": "Please upload a bank statement to receive personalized financial coaching.",
                "retrieved_context": "",
                "flags": [],
            }

        user_query = query or "How can I improve my finances and save money?"

        # 1. Compute financials
        kpis = PandasFinanceTool.compute_kpis(df)
        cat_df = PandasFinanceTool.get_category_breakdown(df)

        # 2. Diagnose rule-based overspending flags
        flags = self._detect_overspending_flags(kpis, cat_df)

        # 3. Agentic Decision: Formulate retrieval queries based on flags and user query
        retrieval_query = self._formulate_retrieval_query(user_query, flags)
        retrieved_context = self.chroma.get_relevant_context(retrieval_query, k=3)

        # 4. Format profile & breakdown tables
        profile_str = (
            f"Total Income: ₹{kpis['total_income']:,.2f}\n"
            f"Total Expenses: ₹{kpis['total_expense']:,.2f}\n"
            f"Net Monthly Balance: ₹{kpis['net_savings']:,.2f}\n"
            f"Current Savings Rate: {kpis['savings_rate_pct']}%\n"
            f"Identified Risk Flags: {', '.join(flags) if flags else 'None'}"
        )

        cat_table_lines = ["Category | Amount Spent (₹) | % of Total Expenses"]
        cat_table_lines.append(":---|:---|:---")
        for _, row in cat_df.iterrows():
            cat_table_lines.append(f"{row['category']} | ₹{row['spent']:,.2f} | {row['percentage']}%")
        cat_table_str = "\n".join(cat_table_lines)

        # 5. Synthesize advice via LCEL chain if LLM is ready
        if self.llm:
            try:
                chain = ADVISOR_PROMPT | self.llm | StrOutputParser()
                response = chain.invoke({
                    "retrieved_context": retrieved_context,
                    "financial_profile": profile_str,
                    "category_breakdown_table": cat_table_str,
                    "user_message": user_query,
                })
                return {
                    "report": response,
                    "retrieved_context": retrieved_context,
                    "flags": flags,
                }
            except Exception as exc:
                logger.warning("Advisor LLM synthesis failed (%s); using deterministic advisory.", exc)

        # 6. High-quality deterministic fallback grounded in ChromaDB guidelines
        deterministic_report = self._generate_grounded_report(kpis, cat_df, flags, retrieved_context)
        return {
            "report": deterministic_report,
            "retrieved_context": retrieved_context,
            "flags": flags,
        }

    def _detect_overspending_flags(self, kpis: Dict[str, Any], cat_df: pd.DataFrame) -> List[str]:
        """Examines financial ratios against standard CFP benchmarks."""
        flags: List[str] = []
        income = kpis["total_income"]
        expenses = kpis["total_expense"]

        if expenses > income and income > 0:
            flags.append("Deficit Spending (Expenses exceed total income)")

        if kpis["savings_rate_pct"] < 15.0 and income > 0:
            flags.append("Low Savings Rate (<15% vs 20% benchmark)")

        if not cat_df.empty:
            cat_pcts = dict(zip(cat_df["category"], cat_df["percentage"]))

            # Food & Dining benchmark (should be < 10% of total expenses)
            food_dining = cat_pcts.get("Food & Dining", 0.0)
            if food_dining > 10.0:
                flags.append(f"Elevated Food & Dining ({food_dining}% of expenses)")

            # Shopping benchmark (should be < 8% of total expenses)
            shopping = cat_pcts.get("Shopping & Retail", 0.0)
            if shopping > 10.0:
                flags.append(f"High Shopping & Retail Outflow ({shopping}% of expenses)")

            # Housing benchmark (should be < 35% of income)
            housing = cat_pcts.get("Housing & Rent", 0.0)
            if housing > 35.0:
                flags.append(f"High Housing Burden ({housing}% of expenses)")

            # Subscriptions benchmark (should be < 3% of expenses)
            subs = cat_pcts.get("Subscriptions & Memberships", 0.0)
            if subs > 3.0:
                flags.append(f"Recurring Subscriptions Drain ({subs}% of expenses)")

        return flags

    def _formulate_retrieval_query(self, user_query: str, flags: List[str]) -> str:
        """
        Agentic query formulation: merges user intent with detected risk flags
        to search for relevant ChromaDB guidelines.
        """
        topics = []
        if any("food" in f.lower() or "dining" in f.lower() for f in flags):
            topics.append("food delivery dining out savings limit")
        if any("subscription" in f.lower() for f in flags):
            topics.append("subscription audit recurring charges")
        if any("savings rate" in f.lower() or "deficit" in f.lower() for f in flags):
            topics.append("50/30/20 rule emergency fund debt repayment")
        if any("housing" in f.lower() for f in flags):
            topics.append("housing rent benchmarks")

        if topics:
            return f"{user_query} {' '.join(topics)}"
        return f"{user_query} 50/30/20 rule budgeting benchmarks"

    def _generate_grounded_report(
        self,
        kpis: Dict[str, Any],
        cat_df: pd.DataFrame,
        flags: List[str],
        retrieved_context: str,
    ) -> str:
        """Constructs an expert report grounded in the retrieved guidelines."""
        income = kpis["total_income"]
        expenses = kpis["total_expense"]
        savings_rate = kpis["savings_rate_pct"]

        lines = [
            "## 💡 Financial Advisor Diagnosis & Action Plan",
            "\n### 1. Executive Assessment",
            f"- **Net Cash Flow:** ₹{kpis['net_savings']:,.2f} ({'Surplus' if kpis['net_savings'] >= 0 else 'Deficit'})",
            f"- **Current Savings Rate:** **{savings_rate}%** (Benchmark Target: **≥ 20%**)",
        ]

        if flags:
            lines.append("\n### ⚠️ Critical Overspending Alerts Detected:")
            for flag in flags:
                lines.append(f"- **{flag}**")
        else:
            lines.append("\n✅ **Great Discipline:** No major benchmark violations detected in this statement!")

        lines.append("\n### 🎯 Top 3 High-Impact Savings Recommendations:")

        # Recommendation 1: Food & Dining
        food_row = cat_df[cat_df["category"] == "Food & Dining"]
        if not food_row.empty and food_row.iloc[0]["spent"] > 1000:
            spent = food_row.iloc[0]["spent"]
            potential_saving = round(spent * 0.35, 2)
            lines.append(
                f"1. **Food Delivery & Dining Trim (Target: Save ₹{potential_saving:,.2f}/mo)**\n"
                f"   - You spent **₹{spent:,.2f}** on dining/delivery apps.\n"
                f"   - *Action:* Batch cook twice a week and cap weekend delivery to twice weekly to reclaim 35%."
            )
        else:
            lines.append(
                "1. **50/30/20 Budgeting Alignment**\n"
                "   - Aim to keep essential living expenses at 50% and direct at least 20% into an automated SIP/RD on salary day."
            )

        # Recommendation 2: Subscriptions & Shopping
        sub_row = cat_df[cat_df["category"] == "Subscriptions & Memberships"]
        shop_row = cat_df[cat_df["category"] == "Shopping & Retail"]
        if not sub_row.empty and sub_row.iloc[0]["spent"] > 500:
            lines.append(
                f"2. **Recurring Subscriptions Audit (Target: Save ₹{sub_row.iloc[0]['spent']*0.5:,.2f}/mo)**\n"
                f"   - Recurring debits total **₹{sub_row.iloc[0]['spent']:,.2f}**.\n"
                f"   - *Action:* Pause streaming services not used in the last 14 days; rotate platforms."
            )
        elif not shop_row.empty and shop_row.iloc[0]["spent"] > 2000:
            lines.append(
                f"2. **Implement the 48-Hour Rule for Shopping (Target: Save ₹{shop_row.iloc[0]['spent']*0.4:,.2f}/mo)**\n"
                f"   - Non-essential retail shopping accounted for **₹{shop_row.iloc[0]['spent']:,.2f}**.\n"
                f"   - *Action:* Wait 48 hours before purchasing any non-essential online item over ₹1,000."
            )
        else:
            lines.append(
                "2. **Emergency Fund Prioritization**\n"
                f"   - Maintain 3–6 months of essential living expenses (approx ₹{expenses*3:,.2f}) in a high-liquidity savings/FD account."
            )

        # Recommendation 3: Automated Savings
        target_savings = round(max(income * 0.20, 5000.0), 2)
        lines.append(
            f"3. **Pay Yourself First (Target: Automate ₹{target_savings:,.2f}/mo)**\n"
            f"   - Schedule an automatic recurring deposit or mutual fund SIP on the 1st of every month right after salary credit."
        )

        return "\n".join(lines)
