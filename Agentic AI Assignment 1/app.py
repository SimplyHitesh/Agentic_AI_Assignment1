"""
app.py — Statement-to-Insights Finance Agent
=============================================
Main Streamlit application entry point for Assignment 4:
Multi-Agent Personal Finance Advisor powered by LangChain, ChromaDB, and OpenAI.

Architecture & Agent Squad:
  1. Parser Agent: Reads uploaded statement (CSV/PDF) and cleans into Pandas DataFrame.
  2. Categorizer Agent: Tags transactions into spending categories using CoT few-shot prompting & rules.
  3. Analysis Agent: Computes totals, trends, and category breakdowns using Pandas and Calculator tools.
  4. Advisor Agent: Agentic RAG retrieving budgeting benchmarks from ChromaDB to flag overspending & advise.
  5. Intent Router: LCEL pipeline classifying queries into discrete intents and dispatching to agents.
"""

from __future__ import annotations

import os
import streamlit as st
import pandas as pd

# ── Local Package Imports ────────────────────────────────────────────────────
from finance_agent.agents.parser_agent import ParserAgent
from finance_agent.agents.categorizer_agent import CategorizerAgent
from finance_agent.agents.analysis_agent import AnalysisAgent
from finance_agent.agents.advisor_agent import AdvisorAgent
from finance_agent.utils.chroma_store import ChromaStore
from finance_agent.utils.router import IntentRouter
from finance_agent.ui.components import (
    render_sidebar,
    render_file_uploader,
    render_kpi_cards,
    render_interactive_table,
    render_charts,
    render_agent_trace,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Streamlit Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Statement-to-Insights Finance Agent",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agent Initialization & Caching
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_chroma_store(api_key: str):
    """Initializes and caches the ChromaDB vector store."""
    return ChromaStore(api_key=api_key)


def get_agent_system(api_key: str):
    """
    Constructs the multi-agent system with the specified API credentials.
    """
    chroma = get_chroma_store(api_key)
    parser = ParserAgent()
    categorizer = CategorizerAgent(api_key=api_key)
    analyzer = AnalysisAgent(api_key=api_key)
    advisor = AdvisorAgent(chroma_store=chroma, api_key=api_key)
    router = IntentRouter(analysis_agent=analyzer, advisor_agent=advisor, api_key=api_key)

    return {
        "chroma": chroma,
        "parser": parser,
        "categorizer": categorizer,
        "analyzer": analyzer,
        "advisor": advisor,
        "router": router,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initializes session state keys."""
    defaults = {
        "messages": [],
        "parsed_df": None,
        "parse_warnings": [],
        "source_format": "unknown",
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "last_uploaded_name": None,
        "traces": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# 4. Upload & Statement Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_uploaded_statement(uploaded_file, agents: dict) -> None:
    """Parses statement file, categorizes transactions, and saves to state."""
    with st.spinner("📄 Step 1/2: Parsing statement structure with Parser Agent..."):
        parse_res = agents["parser"].parse(uploaded_file)

    if not parse_res.success:
        st.error(f"Failed to parse statement: {parse_res.error}", icon="🚨")
        return

    with st.spinner("🏷️ Step 2/2: Categorizing transactions with Categorizer Agent (CoT)..."):
        categorized_df = agents["categorizer"].categorize(parse_res.df)

    st.session_state["parsed_df"] = categorized_df
    st.session_state["parse_warnings"] = parse_res.warnings
    st.session_state["source_format"] = parse_res.source_format
    st.session_state["last_uploaded_name"] = uploaded_file.name

    st.toast(f"Successfully processed {len(categorized_df)} transactions!", icon="✅")

    # Add welcome message to chat if empty
    if not st.session_state["messages"]:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": (
                f"👋 I have analyzed your statement **{uploaded_file.name}** "
                f"({len(categorized_df)} transactions processed).\n\n"
                "You can now ask me:\n"
                "- *'How much did I spend on Food & Dining?'*\n"
                "- *'What are my top 5 largest expenses?'*\n"
                "- *'Where can I save money and reduce overspending?'*\n"
                "- *'Show all payments to Zomato'*"
            ),
        })


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Application Flow
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    init_session_state()

    # Get multi-agent squad
    api_key = st.session_state.get("openai_api_key", "")
    agents = get_agent_system(api_key)

    # Sidebar
    render_sidebar(chroma_store=agents["chroma"])

    # Header
    st.title("💳 Statement-to-Insights Finance Agent")
    st.caption("Autonomous Multi-Agent Personal Finance Advisor · Assignment 4")
    st.divider()

    # ── Statement Upload Section ──────────────────────────────────────────────
    uploaded_file = render_file_uploader()

    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            process_uploaded_statement(uploaded_file, agents)

    # ── Financial Dashboard (when DataFrame is loaded) ────────────────────────
    df = st.session_state.get("parsed_df")

    if df is not None and not df.empty:
        # High Level KPI Cards
        render_kpi_cards(df)

        st.divider()

        # Two-column layout: Left = Table, Right = Visualizations
        col_left, col_right = st.columns([1, 1], gap="medium")

        with col_left:
            render_interactive_table(df, st.session_state.get("parse_warnings", []))

        with col_right:
            render_charts(df)

        st.divider()

    else:
        # Prompt user to upload or use sample statement
        st.info("👆 Upload your bank statement above (CSV or PDF) to begin analysis.", icon="📂")
        if st.button("📁 Load Included Sample Statement (sample_statement.csv)"):
            sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_statement.csv")
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    process_uploaded_statement(f, agents)
                st.rerun()

    # ── Interactive Multi-Agent Chat Interface ────────────────────────────────
    st.subheader("💬 Ask Your Personal Finance Advisor")

    # Quick prompt pills
    pill_cols = st.columns(4)
    quick_queries = [
        "Where can I save money?",
        "What are my top 5 expenses?",
        "How much did I spend on Food?",
        "What is my savings rate?",
    ]
    selected_quick_query = None
    for i, q in enumerate(quick_queries):
        if pill_cols[i].button(q, key=f"quick_btn_{i}", use_container_width=True):
            selected_quick_query = q

    # Render previous messages
    for i, msg in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            trace = st.session_state["traces"].get(i)
            if trace:
                render_agent_trace(trace)

    # Accept user query from text box or quick pill
    user_query = st.chat_input("Ask about your transactions, spending trends, or savings advice...")
    if selected_quick_query:
        user_query = selected_quick_query

    if user_query:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Route query through multi-agent system
        with st.spinner("🤖 Routing to specialized agent and analyzing..."):
            route_result = agents["router"].route(user_query, df=df)

        # Save trace info
        msg_idx = len(st.session_state["messages"])
        st.session_state["traces"][msg_idx] = {
            "intent": route_result.intent,
            "confidence": route_result.confidence,
            "reasoning": route_result.reasoning,
            "agent_name": route_result.agent_name,
            "flags": route_result.flags,
            "retrieved_context": route_result.retrieved_context,
        }

        # Save and display assistant response
        st.session_state["messages"].append({"role": "assistant", "content": route_result.response})
        with st.chat_message("assistant"):
            st.markdown(route_result.response)
            render_agent_trace(st.session_state["traces"][msg_idx])


if __name__ == "__main__":
    main()
