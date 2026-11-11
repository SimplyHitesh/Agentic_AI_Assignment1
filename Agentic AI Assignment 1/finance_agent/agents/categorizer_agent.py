"""
finance_agent/agents/categorizer_agent.py
-----------------------------------------
Categorizer Agent: Tags each transaction into standard spending categories.
Demonstrates:
  - Few-shot Chain-of-Thought (CoT) Prompting via LangChain LCEL
  - High-precision merchant matching heuristics
  - Graceful fallback for offline / mock usage
"""

from __future__ import annotations

import re
import json
import logging
from typing import Optional, List, Dict, Any

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from finance_agent.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    LLM_TEMPERATURE,
    SPENDING_CATEGORIES,
)
from finance_agent.prompts.prompts import CATEGORIZER_PROMPT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fast Merchant Pattern Mappings (Rule-Based Heuristic Layer)
# ─────────────────────────────────────────────────────────────────────────────

MERCHANT_RULES: Dict[str, List[str]] = {
    "Food & Dining": [
        "zomato", "swiggy", "mcdonald", "starbucks", "kfc", "burger king",
        "domino", "pizza", "cafe", "restaurant", "dine", "barbeque", "subway",
        "chai point", "eatfit", "haldiram", "biryani"
    ],
    "Groceries": [
        "bigbasket", "blinkit", "zepto", "instamart", "dmart", "spencer",
        "nature basket", "more retail", "supermarket", "provision", "kirana", "grocer"
    ],
    "Housing & Rent": [
        "rent", "house rent", "landlord", "nobroker", "nestaway", "maintenance fee",
        "society maintenance"
    ],
    "Utilities & Bills": [
        "bescom", "tneb", "electricity", "water supply", "gas", "indane", "hp gas",
        "bharat gas", "broadband", "bsnl", "airtel broadband", "act fibernet", "jio fiber",
        "recharge", "postpaid", "dth", "tata play"
    ],
    "Transportation": [
        "uber", "ola", "rapido", "metro", "irctc", "railway", "fuel", "petrol",
        "diesel", "hpcl", "bpcl", "iocl", "shell", "fastag", "toll"
    ],
    "Shopping & Retail": [
        "amazon", "flipkart", "myntra", "ajio", "zara", "h&m", "ikea", "croma",
        "reliance digital", "tata cliq", "nykaa", "shoppers stop", "retail", "mart"
    ],
    "Entertainment & Leisure": [
        "bookmyshow", "pvr", "inox", "cinema", "theatre", "movie", "steam",
        "playstation", "gaming", "amusement", "bowling"
    ],
    "Healthcare & Medical": [
        "medplus", "apollo", "1mg", "pharmeasy", "hospital", "clinic", "pharmacy",
        "doctor", "diagnostic", "pathology", "dentist", "eyecare"
    ],
    "Education & Courses": [
        "coursera", "udemy", "upgrad", "unacademy", "school", "college", "tuition",
        "university", "exam fee", "books"
    ],
    "Travel & Vacation": [
        "makemytrip", "goibibo", "yatra", "cleartrip", "indigo", "air india",
        "spicejet", "hotel", "resort", "airbnb", "booking.com"
    ],
    "Subscriptions & Memberships": [
        "netflix", "spotify", "prime video", "youtube premium", "hotstar", "disney",
        "apple.com", "google storage", "cloud storage", "gym", "cult.fit"
    ],
    "Financial & Loan EMI": [
        "emi", "loan", "home loan", "car loan", "hdfc loan", "sbi loan", "bajaj finance",
        "credit card bill", "nach", "insurance", "lic", "hdfc life", "sbi life"
    ],
    "Income & Salary": [
        "salary", "payroll", "dividend", "interest credited", "stipend", "bonus"
    ],
}


class CategorizerAgent:
    """
    Tags bank transactions with spending categories using a hybrid approach:
    1. Fast heuristic keyword matching against known merchants.
    2. LangChain Chain-of-Thought (CoT) few-shot LLM categorization for ambiguous rows.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.llm = self._init_llm()

    def _init_llm(self) -> Optional[ChatOpenAI]:
        """Initialize LangChain ChatOpenAI if credentials exist."""
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
            logger.warning("Failed to initialize LLM in CategorizerAgent: %s", e)
            return None

    @classmethod
    def match_rule(cls, text: str, amount: float, txn_type: str) -> Optional[str]:
        """Matches transaction narration against regex/keyword rules."""
        text_lower = text.lower()

        # Check Income first if it is a credit
        if txn_type == "credit":
            for kw in MERCHANT_RULES["Income & Salary"]:
                if kw in text_lower:
                    return "Income & Salary"

        for category, keywords in MERCHANT_RULES.items():
            for kw in keywords:
                # Word boundary or containment
                if re.search(r"\b" + re.escape(kw) + r"\b", text_lower) or kw in text_lower:
                    return category

        # Fallback for credits
        if txn_type == "credit":
            return "Income & Salary" if amount > 5000 else "Transfers"

        return None

    def categorize(self, df: pd.DataFrame, use_llm: bool = True) -> pd.DataFrame:
        """
        Categorizes transactions in the DataFrame.
        Adds 'category' and 'categorization_method' columns.
        """
        if df.empty:
            return df

        df = df.copy()
        categories: List[str] = []
        methods: List[str] = []
        ambiguous_indices: List[int] = []

        # ── 1. Pass 1: Fast Rule-based Categorization ─────────────────────────
        for idx, row in df.iterrows():
            desc = str(row.get("description", ""))
            amount = float(row.get("amount", 0.0))
            txn_type = str(row.get("transaction_type", "debit"))

            rule_cat = self.match_rule(desc, amount, txn_type)
            if rule_cat:
                categories.append(rule_cat)
                methods.append("rule")
            else:
                categories.append("Other")
                methods.append("unmatched")
                ambiguous_indices.append(idx)

        df["category"] = categories
        df["categorization_method"] = methods

        # ── 2. Pass 2: LangChain CoT LLM for Ambiguous Transactions ──────────
        if use_llm and self.llm and ambiguous_indices:
            df = self._categorize_with_llm(df, ambiguous_indices)

        return df

    def _categorize_with_llm(self, df: pd.DataFrame, indices: List[int]) -> pd.DataFrame:
        """
        Uses LangChain LCEL pipeline to categorize ambiguous items in batches.
        """
        batch_limit = min(len(indices), 15)  # Cap per batch to control latency
        target_indices = indices[:batch_limit]

        batch_lines = []
        for i, idx in enumerate(target_indices):
            row = df.loc[idx]
            batch_lines.append(
                f"[{i}] Narration: '{row['description']}' | Amount: {row['amount']} | Type: {row['transaction_type']}"
            )

        transactions_text = "\n".join(batch_lines)

        try:
            # Build LCEL chain: PromptTemplate | ChatOpenAI | StrOutputParser
            chain = CATEGORIZER_PROMPT | self.llm | StrOutputParser()
            response_text = chain.invoke({"transactions_text": transactions_text})

            # Clean JSON markdown blocks if LLM wrapped in ```json
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed_list = json.loads(cleaned)
            for item in parsed_list:
                item_idx = item.get("index")
                cat = item.get("category")
                if item_idx is not None and 0 <= item_idx < len(target_indices):
                    target_row_idx = target_indices[item_idx]
                    if cat in SPENDING_CATEGORIES:
                        df.at[target_row_idx, "category"] = cat
                        df.at[target_row_idx, "categorization_method"] = "llm_cot"
        except Exception as exc:
            logger.warning("LLM Categorization failed, retaining rule defaults: %s", exc)

        return df
