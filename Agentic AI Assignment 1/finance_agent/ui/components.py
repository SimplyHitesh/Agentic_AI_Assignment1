"""
finance_agent/ui/components.py
--------------------------------
Reusable Streamlit UI components for the Statement-to-Insights Finance Agent.
Provides:
  - Sidebar with API Configuration & ChromaDB Status
  - KPI Metric Cards with Inflow, Outflow, Net Balance, and Savings Rate
  - Interactive Filterable Transaction Table
  - Visual Analytics: Category Donut, Monthly Trend, Top Expenses
  - Multi-Agent Reasoning Trace Expander (showing intent, routed agent, and RAG context)
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, List, Dict, Any

from finance_agent.tools.pandas_tool import PandasFinanceTool


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Component
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(chroma_store=None) -> None:
    """Renders configuration, API keys, and system diagnostics."""
    with st.sidebar:
        st.markdown("## 💳 Finance Agent")
        st.caption("Multi-Agent Statement-to-Insights · LangChain + ChromaDB")
        st.divider()

        # ── LLM Configuration ─────────────────────────────────────────────
        st.subheader("⚙️ LLM & Provider")
        
        current_key = st.session_state.get("openai_api_key", "")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=current_key,
            placeholder="sk-proj-...",
            help="Your key is held only in current session memory.",
        )
        if api_key_input != current_key:
            st.session_state["openai_api_key"] = api_key_input

        # Advanced Settings Expander
        with st.expander("🔧 Advanced LLM Settings"):
            selected_model = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "custom/local"],
                index=0,
            )
            st.session_state["selected_model"] = selected_model

            base_url = st.text_input(
                "Custom Base URL (Local LLM / Ollama)",
                value=st.session_state.get("openai_base_url", ""),
                placeholder="http://localhost:11434/v1",
                help="Leave empty for official OpenAI endpoints.",
            )
            st.session_state["openai_base_url"] = base_url

        st.divider()

        # ── ChromaDB Vector Store Status ──────────────────────────────────
        st.subheader("📚 ChromaDB Knowledge Base")
        doc_count = chroma_store.count() if chroma_store else 0
        st.success(f"Indexed Guidelines: **{doc_count} chunks**", icon="🗄️")
        with st.expander("View Indexed Guidelines Topics"):
            st.markdown(
                """
                - **50/30/20 Budgeting Rule**
                - **Category Spending Thresholds**
                - **Emergency Fund Formulas**
                - **Subscription & Dining Optimisation**
                - **Debt Snowball vs Avalanche**
                """
            )

        st.divider()

        # ── Reset / Action Buttons ────────────────────────────────────────
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.caption("Agentic AI Assignment 4 · Statement-to-Insights Agent")


# ─────────────────────────────────────────────────────────────────────────────
# File Uploader
# ─────────────────────────────────────────────────────────────────────────────

def render_file_uploader():
    """Renders the file uploader for CSV and PDF statements."""
    uploaded = st.file_uploader(
        label="Upload Bank or UPI Statement",
        type=["csv", "pdf"],
        help="Upload CSV or PDF statements from SBI, HDFC, ICICI, Paytm, PhonePe, or standard bank exports.",
        accept_multiple_files=False,
    )
    return uploaded


# ─────────────────────────────────────────────────────────────────────────────
# KPI Metric Cards
# ─────────────────────────────────────────────────────────────────────────────

def render_kpi_cards(df: pd.DataFrame) -> None:
    """Renders 4 top-level financial KPI metric cards."""
    kpis = PandasFinanceTool.compute_kpis(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="💰 Total Inflow (Income)",
        value=f"₹{kpis['total_income']:,.0f}",
        delta=f"{kpis['transaction_count']} transactions",
        delta_color="off",
    )
    col2.metric(
        label="💸 Total Outflow (Expenses)",
        value=f"₹{kpis['total_expense']:,.0f}",
        delta=f"Avg: ₹{kpis['avg_debit']:,.0f}",
        delta_color="inverse",
    )
    net_val = kpis["net_savings"]
    col3.metric(
        label="⚖️ Net Cash Flow",
        value=f"₹{net_val:,.0f}",
        delta="Surplus" if net_val >= 0 else "Deficit",
        delta_color="normal" if net_val >= 0 else "inverse",
    )
    rate = kpis["savings_rate_pct"]
    col4.metric(
        label="📈 Savings Rate",
        value=f"{rate:.1f}%",
        delta="Target: ≥20%",
        delta_color="normal" if rate >= 20 else "inverse",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Data Table
# ─────────────────────────────────────────────────────────────────────────────

def render_interactive_table(df: pd.DataFrame, warnings: List[str]) -> None:
    """Renders the categorized transaction table with filtering."""
    for w in warnings:
        st.warning(w, icon="⚠️")

    with st.expander("📋 View & Filter Categorized Transactions", expanded=False):
        filter_col1, filter_col2 = st.columns([2, 1])

        with filter_col1:
            search_query = st.text_input("🔍 Search Description or Merchant", placeholder="e.g. Zomato, Uber, Rent")

        with filter_col2:
            categories = ["All"] + sorted(list(df["category"].dropna().unique()))
            selected_cat = st.selectbox("Category Filter", categories)

        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["description"].str.contains(search_query, case=False, na=False)
            ]
        if selected_cat != "All":
            filtered_df = filtered_df[filtered_df["category"] == selected_cat]

        # Format for clean display
        disp = filtered_df.copy()
        disp["date"] = disp["date"].dt.strftime("%d %b %Y")
        disp["amount_formatted"] = disp["amount"].apply(
            lambda x: f"₹{x:,.2f}" if x >= 0 else f"-₹{abs(x):,.2f}"
        )

        display_cols = ["date", "description", "category", "amount_formatted", "transaction_type"]
        if "categorization_method" in disp.columns:
            display_cols.append("categorization_method")

        st.dataframe(
            disp[display_cols].rename(columns={
                "date": "Date",
                "description": "Description",
                "category": "Category",
                "amount_formatted": "Amount (₹)",
                "transaction_type": "Type",
                "categorization_method": "Tag Method",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Visual Charts
# ─────────────────────────────────────────────────────────────────────────────

def render_charts(df: pd.DataFrame) -> None:
    """Renders interactive Plotly visualisations."""
    st.subheader("📊 Visual Analytics")

    debits_df = df[df["transaction_type"] == "debit"].copy()
    if debits_df.empty:
        st.info("No debit transactions found to visualize.")
        return

    cat_breakdown = PandasFinanceTool.get_category_breakdown(df)

    tab1, tab2, tab3 = st.tabs(["Spending by Category", "Top 5 Expenses", "Monthly Trend"])

    with tab1:
        # Donut Chart for Categories
        if not cat_breakdown.empty:
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=cat_breakdown["category"],
                        values=cat_breakdown["spent"],
                        hole=0.45,
                        textinfo="label+percent",
                        hoverinfo="label+value+percent",
                    )
                ]
            )
            fig_donut.update_layout(
                title_text="Expense Distribution by Category",
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No categorized expense data.")

    with tab2:
        # Top 5 Expenses Bar Chart
        top5 = PandasFinanceTool.get_top_transactions(df, n=5)
        if not top5.empty:
            top5["short_desc"] = top5["description"].apply(
                lambda s: s[:25] + "..." if len(s) > 25 else s
            )
            fig_top5 = px.bar(
                top5,
                x="short_desc",
                y=top5["amount"].abs(),
                color="category",
                labels={"short_desc": "Transaction", "y": "Amount (₹)"},
                title="Top 5 Largest Debits",
                text_auto=".2s",
            )
            fig_top5.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_top5, use_container_width=True)

    with tab3:
        # Monthly Income vs Expense
        monthly = PandasFinanceTool.get_monthly_breakdown(df)
        if not monthly.empty:
            fig_monthly = go.Figure()
            fig_monthly.add_trace(
                go.Bar(name="Income", x=monthly["month"], y=monthly["income"], marker_color="#2ca02c")
            )
            fig_monthly.add_trace(
                go.Bar(name="Expenses", x=monthly["month"], y=monthly["expense"], marker_color="#d62728")
            )
            fig_monthly.update_layout(
                barmode="group",
                title="Monthly Cash Inflow vs Outflow",
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
            )
            st.plotly_chart(fig_monthly, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Trace / Reasoning Viewer
# ─────────────────────────────────────────────────────────────────────────────

def render_agent_trace(trace: Dict[str, Any]) -> None:
    """Renders a collapsible trace showing multi-agent reasoning steps."""
    with st.expander("🔍 Multi-Agent Execution Trace & RAG Context", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.write(f"**Detected Intent:** `{trace.get('intent', 'N/A')}`")
        col2.write(f"**Confidence:** `{trace.get('confidence', 0.0):.2f}`")
        col3.write(f"**Dispatched Agent:** `{trace.get('agent_name', 'N/A')}`")

        st.markdown(f"**Routing CoT Reasoning:** {trace.get('reasoning', 'Direct dispatch')}")

        if trace.get("flags"):
            st.warning(f"**Risk Flags Identified:** {', '.join(trace['flags'])}")

        if trace.get("retrieved_context"):
            st.markdown("#### 📖 Retrieved ChromaDB Context (Agentic RAG):")
            st.markdown(trace["retrieved_context"])
