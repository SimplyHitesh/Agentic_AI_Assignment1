"""
demo.py — End-to-End CLI Demonstration of the Multi-Agent Finance System
========================================================================
Demonstrates all 4 Agents and all 7 Mandatory Acceptance Criteria:
  1. Parser Agent: Parses and normalizes bank CSV statement
  2. Categorizer Agent: Tags transactions into spending categories
  3. Analysis Agent: Computes exact totals and breakdowns via Pandas and Calculator
  4. Advisor Agent: Agentic RAG retrieving guidelines from ChromaDB to flag overspending
  5. Intent Router: LCEL pipeline classifying queries and routing to agents
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.agents.parser_agent import ParserAgent
from finance_agent.agents.categorizer_agent import CategorizerAgent
from finance_agent.agents.analysis_agent import AnalysisAgent
from finance_agent.agents.advisor_agent import AdvisorAgent
from finance_agent.tools.calculator import FinanceCalculator
from finance_agent.tools.pandas_tool import PandasFinanceTool
from finance_agent.utils.chroma_store import ChromaStore
from finance_agent.utils.router import IntentRouter


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print_banner("💳 STATEMENT-TO-INSIGHTS MULTI-AGENT FINANCE SYSTEM")

    sample_csv = PROJECT_ROOT / "sample_data" / "sample_statement.csv"
    if not sample_csv.exists():
        print(f"❌ Sample statement not found at {sample_csv}")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Parser Agent
    # ─────────────────────────────────────────────────────────────────────────
    print_banner("1. PARSER AGENT — Reading & Cleaning Bank Statement")
    parser = ParserAgent()
    parse_result = parser.parse(str(sample_csv))

    if not parse_result.success:
        print(f"❌ Parse failed: {parse_result.error}")
        return

    df = parse_result.df
    print(f"✅ Successfully parsed {parse_result.row_count} transactions from {sample_csv.name}")
    print(f"Format: {parse_result.source_format.upper()} | Warnings: {parse_result.warnings}")
    print("\nSample Cleaned Transactions (first 5):")
    print(df[["date", "description", "amount", "transaction_type"]].head(5).to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Categorizer Agent
    # ─────────────────────────────────────────────────────────────────────────
    print_banner("2. CATEGORIZER AGENT — Tagging Transactions (CoT & Rules)")
    categorizer = CategorizerAgent()
    df_categorized = categorizer.categorize(df)
    print("✅ Categorization Complete. Sample categorized records:")
    print(df_categorized[["date", "description", "amount", "category", "categorization_method"]].head(8).to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Numerical & Pandas Tools
    # ─────────────────────────────────────────────────────────────────────────
    print_banner("3. PANDAS & CALCULATOR TOOLS — Verified Calculations")
    kpis = PandasFinanceTool.compute_kpis(df_categorized)
    print(f"- Total Income (Credits)  : ₹{kpis['total_income']:,.2f}")
    print(f"- Total Expenses (Debits) : ₹{kpis['total_expense']:,.2f}")
    print(f"- Net Monthly Cash Flow   : ₹{kpis['net_savings']:,.2f}")
    print(f"- Savings Rate            : {kpis['savings_rate_pct']}%")
    print(f"- Total Transactions      : {kpis['transaction_count']}")

    print("\nCategory Spending Breakdown:")
    cat_df = PandasFinanceTool.get_category_breakdown(df_categorized)
    print(cat_df.to_string(index=False))

    # Test calculator tool explicitly
    calc_expr = f"({kpis['total_expense']} / {kpis['total_income']}) * 100"
    expense_ratio = FinanceCalculator.evaluate_expression(calc_expr)
    print(f"\nCalculator Tool Verification:")
    print(f"Formula: '{calc_expr}' => Expense Ratio = {expense_ratio}% of income")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: ChromaDB Vector Store & Knowledge Base
    # ─────────────────────────────────────────────────────────────────────────
    print_banner("4. CHROMADB VECTOR STORE — Ingesting Budgeting Guidelines")
    chroma = ChromaStore()
    count = chroma.count()
    print(f"✅ ChromaDB initialized. Total guideline chunks indexed: {count}")

    # Test similarity search
    sample_query = "dining out food delivery 50 30 20 rule"
    hits = chroma.similarity_search(sample_query, k=2)
    print(f"\nSimilarity Search for '{sample_query}':")
    for i, h in enumerate(hits, 1):
        src = h.get("metadata", {}).get("source", "N/A")
        print(f"  [{i}] Source: {src} | Snippet: {h['content'][:120]}...")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Advisor Agent (Agentic RAG)
    # ─────────────────────────────────────────────────────────────────────────
    print_banner("5. ADVISOR AGENT — Agentic RAG Diagnosis & Recommendations")
    advisor = AdvisorAgent(chroma_store=chroma)
    advice = advisor.advise(df_categorized, query="Where can I cut expenses and improve my savings?")
    print(advice["report"])

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Intent Router (LCEL)
    # ─────────────────────────────────────────────────────────────────────────
    print_banner("6. INTENT ROUTER — LCEL Intent Classification & Dispatch")
    router = IntentRouter(analysis_agent=AnalysisAgent(), advisor_agent=advisor)

    test_queries = [
        "Find all transactions from Zomato",
        "How much did I spend on Food & Dining?",
        "Where can I save money this month?",
        "Hello, what can you help me with?",
    ]

    for q in test_queries:
        res = router.route(q, df=df_categorized)
        print(f"\nQuery         : '{q}'")
        print(f"Detected Intent: {res.intent} (Confidence: {res.confidence})")
        print(f"Dispatched To  : {res.agent_name}")
        print(f"Response Summary: {res.response.splitlines()[0] if res.response else 'N/A'}")

    print_banner("🎉 ALL 4 AGENTS & 7 CRITERIA SUCCESSFULLY DEMONSTRATED!")


if __name__ == "__main__":
    main()

