"""
finance_agent/utils/router.py
-----------------------------
Intent Detection and Agent Routing using LangChain Expression Language (LCEL).
Fulfills Mandatory Criteria 4 & 6:
  - Classifies user queries into discrete financial intents.
  - Connects prompts, LLM, and output processing with LCEL.
  - Dynamically routes to the corresponding agent or tool.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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
from finance_agent.prompts.prompts import INTENT_ROUTER_PROMPT
from finance_agent.tools.pandas_tool import PandasFinanceTool
from finance_agent.agents.analysis_agent import AnalysisAgent
from finance_agent.agents.advisor_agent import AdvisorAgent

logger = logging.getLogger(__name__)

# Canonical Intent Identifiers
INTENT_TRANSACTION_LOOKUP = "transaction_lookup"
INTENT_SPENDING_ANALYSIS = "spending_analysis"
INTENT_SAVINGS_ADVICE = "savings_advice"
INTENT_GENERAL_FAQ = "general_faq"

ALL_INTENTS = [
    INTENT_TRANSACTION_LOOKUP,
    INTENT_SPENDING_ANALYSIS,
    INTENT_SAVINGS_ADVICE,
    INTENT_GENERAL_FAQ,
]


@dataclass
class RouteResult:
    """Structured response returned by IntentRouter."""
    intent: str
    confidence: float
    reasoning: str
    agent_name: str
    response: str
    retrieved_context: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentRouter:
    """
    Orchestrates the multi-agent system by classifying user intent and dispatching
    to the appropriate specialized agent.
    """

    def __init__(
        self,
        analysis_agent: Optional[AnalysisAgent] = None,
        advisor_agent: Optional[AdvisorAgent] = None,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or OPENAI_API_KEY
        self.analysis_agent = analysis_agent or AnalysisAgent(api_key=self.api_key)
        self.advisor_agent = advisor_agent or AdvisorAgent(api_key=self.api_key)
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
            logger.warning("Could not initialize LLM for IntentRouter: %s", e)
            return None

    def detect_intent_lcel(self, query: str) -> Dict[str, Any]:
        """
        Uses LangChain Expression Language (LCEL) chain to classify intent.
        Pipeline: INTENT_ROUTER_PROMPT | ChatOpenAI | StrOutputParser | JSON Parser
        """
        if not self.llm:
            return self._detect_intent_heuristic(query)

        try:
            # Build LCEL chain
            chain = INTENT_ROUTER_PROMPT | self.llm | StrOutputParser()
            raw_response = chain.invoke({"user_message": query})

            # Clean potential JSON code fences
            clean_str = raw_response.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            clean_str = clean_str.strip()

            parsed = json.loads(clean_str)
            intent = parsed.get("intent", INTENT_SPENDING_ANALYSIS)
            if intent not in ALL_INTENTS:
                intent = INTENT_SPENDING_ANALYSIS

            return {
                "intent": intent,
                "confidence": float(parsed.get("confidence", 0.9)),
                "reasoning": parsed.get("reasoning", "Classified via LCEL prompt."),
                "extracted_entities": parsed.get("extracted_entities", {}),
            }
        except Exception as exc:
            logger.warning("LCEL Intent detection failed (%s); using heuristic.", exc)
            return self._detect_intent_heuristic(query)

    def _detect_intent_heuristic(self, query: str) -> Dict[str, Any]:
        """High-precision keyword fallback when running offline or without API key."""
        q = query.lower()

        # Savings advice keywords
        advice_kw = ["save", "saving", "advice", "suggest", "cut", "reduce", "budget", "overspend", "guideline", "plan", "habit"]
        if any(w in q for w in advice_kw):
            return {
                "intent": INTENT_SAVINGS_ADVICE,
                "confidence": 0.88,
                "reasoning": "Keyword match on financial advice/budgeting terms.",
                "extracted_entities": {},
            }

        # Transaction lookup keywords
        lookup_kw = ["find", "search", "show me transaction", "list transaction", "did i pay", "who is", "when did i", "filter"]
        if any(w in q for w in lookup_kw):
            return {
                "intent": INTENT_TRANSACTION_LOOKUP,
                "confidence": 0.85,
                "reasoning": "Keyword match on transaction search/lookup terms.",
                "extracted_entities": {},
            }

        # General / greeting keywords
        faq_kw = ["hello", "hi", "hey", "who are you", "help", "how do you work", "what can you do"]
        if any(q.strip() == w or q.startswith(w + " ") for w in faq_kw):
            return {
                "intent": INTENT_GENERAL_FAQ,
                "confidence": 0.95,
                "reasoning": "Greeting or general inquiry detected.",
                "extracted_entities": {},
            }

        # Default to spending analysis
        return {
            "intent": INTENT_SPENDING_ANALYSIS,
            "confidence": 0.75,
            "reasoning": "Defaulting to spending quantitative analysis.",
            "extracted_entities": {},
        }

    def route(self, user_message: str, df: Optional[pd.DataFrame] = None) -> RouteResult:
        """
        Dispatches user message to the appropriate agent based on classified intent.
        """
        classification = self.detect_intent_lcel(user_message)
        intent = classification["intent"]
        confidence = classification["confidence"]
        reasoning = classification["reasoning"]

        if df is None or df.empty:
            return RouteResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                agent_name="System",
                response="Please upload a statement (CSV or PDF) first so I can analyze your transactions.",
            )

        # ── 1. Route to Transaction Lookup (Parser / Data filter) ─────────────
        if intent == INTENT_TRANSACTION_LOOKUP:
            response_text = self._handle_transaction_lookup(user_message, df)
            return RouteResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                agent_name="ParserAgent / DataFilter",
                response=response_text,
            )

        # ── 2. Route to Analysis Agent ────────────────────────────────────────
        elif intent == INTENT_SPENDING_ANALYSIS:
            analysis_output = self.analysis_agent.analyze(df, user_message)
            return RouteResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                agent_name="AnalysisAgent",
                response=analysis_output,
            )

        # ── 3. Route to Advisor Agent (Agentic RAG) ───────────────────────────
        elif intent == INTENT_SAVINGS_ADVICE:
            advise_result = self.advisor_agent.advise(df, user_message)
            return RouteResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                agent_name="AdvisorAgent (RAG)",
                response=advise_result["report"],
                retrieved_context=advise_result.get("retrieved_context"),
                flags=advise_result.get("flags", []),
            )

        # ── 4. General FAQ / Help ─────────────────────────────────────────────
        else:
            faq_response = (
                "👋 Hello! I am your **Statement-to-Insights Finance Agent**.\n\n"
                "Here is what I can do for you:\n"
                "- 🔍 **Transaction Lookup:** *'Show all transactions from Zomato'* or *'Find payments above ₹2,000'*\n"
                "- 📊 **Spending Analysis:** *'What was my total spending on food?'* or *'Show my top 5 expenses'*\n"
                "- 💡 **Savings Advisory (Agentic RAG):** *'Where can I save money?'* or *'Am I overspending on dining out?'*\n\n"
                "Try asking one of these questions!"
            )
            return RouteResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                agent_name="AssistantFAQ",
                response=faq_response,
            )

    def _handle_transaction_lookup(self, query: str, df: pd.DataFrame) -> str:
        """Handles searches and transaction filtering."""
        words = [w for w in query.replace("?", "").replace(",", "").split() if len(w) > 3]
        results = pd.DataFrame()

        # Try to find matching transactions by word
        for word in words:
            if word.lower() in ["show", "list", "find", "transactions", "statement", "where", "what", "paid"]:
                continue
            matched = df[
                df["description"].str.lower().str.contains(word.lower(), na=False)
                | df.get("category", pd.Series(dtype=str)).str.lower().str.contains(word.lower(), na=False)
            ]
            if not matched.empty:
                results = matched
                break

        if results.empty:
            return f"No specific transactions matched your search query '{query}'. Try searching by merchant name (e.g. 'Zomato', 'BESCOM', 'Uber') or category."

        lines = [f"Found **{len(results)}** matching transaction(s):\n"]
        for _, row in results.head(10).iterrows():
            dt = row["date"].strftime("%d %b %Y") if pd.notnull(row["date"]) else "N/A"
            amt = f"₹{row['amount']:,.2f}" if row['amount'] >= 0 else f"-₹{abs(row['amount']):,.2f}"
            cat = row.get("category", "Other")
            lines.append(f"- **{dt}** | **{amt}** | `{cat}` | {row['description']}")

        return "\n".join(lines)
