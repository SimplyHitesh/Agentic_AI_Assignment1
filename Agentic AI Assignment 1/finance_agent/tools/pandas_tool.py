"""
finance_agent/tools/pandas_tool.py
----------------------------------
Pandas Data Tool for structured aggregations, filtering, and statistical analysis.
Guarantees fast, reproducible numerical summaries from the clean transaction DataFrame.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List


class PandasFinanceTool:
    """Wrapper for DataFrame operations on financial statements."""

    @staticmethod
    def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute high-level Key Performance Indicators:
        - Total Inflow (Credits)
        - Total Outflow (Debits)
        - Net Cash Flow (Inflow - Outflow)
        - Savings Rate %
        - Total Transactions
        - Average Debit Transaction Size
        """
        if df.empty:
            return {
                "total_income": 0.0,
                "total_expense": 0.0,
                "net_savings": 0.0,
                "savings_rate_pct": 0.0,
                "transaction_count": 0,
                "avg_debit": 0.0,
                "date_min": "N/A",
                "date_max": "N/A",
            }

        credits = df.loc[df["transaction_type"] == "credit", "amount"].sum()
        debit_mask = df["transaction_type"] == "debit"
        debits = df.loc[debit_mask, "amount"].abs().sum()
        net = credits - debits
        savings_rate = round((net / credits) * 100.0, 2) if credits > 0 else 0.0
        debit_count = debit_mask.sum()
        avg_debit = round(debits / debit_count, 2) if debit_count > 0 else 0.0

        date_min = df["date"].min().strftime("%d %b %Y") if not df["date"].isna().all() else "N/A"
        date_max = df["date"].max().strftime("%d %b %Y") if not df["date"].isna().all() else "N/A"

        return {
            "total_income": round(float(credits), 2),
            "total_expense": round(float(debits), 2),
            "net_savings": round(float(net), 2),
            "savings_rate_pct": savings_rate,
            "transaction_count": len(df),
            "avg_debit": avg_debit,
            "date_min": date_min,
            "date_max": date_max,
        }

    @staticmethod
    def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates debits by category, providing spent amount, transaction count,
        and percentage of total spending.
        """
        if df.empty or "category" not in df.columns:
            return pd.DataFrame(columns=["category", "spent", "count", "percentage"])

        debits_df = df[df["transaction_type"] == "debit"].copy()
        if debits_df.empty:
            return pd.DataFrame(columns=["category", "spent", "count", "percentage"])

        total_spent = debits_df["amount"].abs().sum()

        grouped = (
            debits_df.groupby("category")
            .agg(
                spent=("amount", lambda s: round(float(s.abs().sum()), 2)),
                count=("amount", "count"),
            )
            .reset_index()
        )

        if total_spent > 0:
            grouped["percentage"] = (grouped["spent"] / total_spent * 100).round(2)
        else:
            grouped["percentage"] = 0.0

        return grouped.sort_values(by="spent", ascending=False).reset_index(drop=True)

    @staticmethod
    def get_top_transactions(
        df: pd.DataFrame, n: int = 5, transaction_type: str = "debit"
    ) -> pd.DataFrame:
        """Returns the top N transactions by absolute amount."""
        if df.empty:
            return pd.DataFrame()

        filtered = df[df["transaction_type"] == transaction_type].copy()
        filtered["abs_amount"] = filtered["amount"].abs()
        top_df = filtered.sort_values(by="abs_amount", ascending=False).head(n)
        return top_df.drop(columns=["abs_amount"])

    @staticmethod
    def filter_transactions(
        df: pd.DataFrame,
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
    ) -> pd.DataFrame:
        """Filters the transactions dataframe by multiple criteria."""
        if df.empty:
            return df

        res = df.copy()

        if keyword:
            kw = keyword.lower().strip()
            res = res[
                res["description"].str.lower().str.contains(kw, na=False)
                | res["raw_text"].str.lower().str.contains(kw, na=False)
            ]

        if category and category.lower() != "all":
            res = res[res["category"].str.lower() == category.lower()]

        if min_amount is not None:
            res = res[res["amount"].abs() >= min_amount]

        if max_amount is not None:
            res = res[res["amount"].abs() <= max_amount]

        return res.reset_index(drop=True)

    @staticmethod
    def get_monthly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
        """Computes monthly credits and debits."""
        if df.empty:
            return pd.DataFrame()

        df_copy = df.copy()
        df_copy["month"] = df_copy["date"].dt.to_period("M").astype(str)

        monthly = (
            df_copy.groupby("month")
            .apply(
                lambda g: pd.Series({
                    "income": round(float(g.loc[g["transaction_type"] == "credit", "amount"].sum()), 2),
                    "expense": round(float(g.loc[g["transaction_type"] == "debit", "amount"].abs().sum()), 2),
                    "count": len(g),
                })
            )
            .reset_index()
        )
        monthly["net"] = (monthly["income"] - monthly["expense"]).round(2)
        return monthly

