# 💳 Statement-to-Insights Finance Agent (Assignment 4)

An autonomous multi-agent personal finance assistant that ingests bank and UPI statements, extracts and categorises transactions, computes spending trends using Pandas and Calculator tools, and produces plain-language insights and overspending alerts via **Agentic RAG** and **ChromaDB**.

---

## 🌟 Key Features & Architecture

```
                                  ┌────────────────────────┐
                                  │   User Web Interface   │
                                  │     (Streamlit UI)     │
                                  └───────────┬────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         [ Statement Upload ]                                  [ Interactive Chat ]
                    │                                                   │
                    ▼                                                   ▼
       ┌────────────────────────┐                          ┌────────────────────────┐
       │      Parser Agent      │                          │     Intent Router      │
       │  (CSV/PDF Data Clean)  │                          │    (LCEL Classifier)   │
       └────────────┬───────────┘                          └────────────┬───────────┘
                    │                                                   │
                    ▼                                      ┌────────────┼────────────┐
       ┌────────────────────────┐                          │            │            │
       │    Categorizer Agent   │                          ▼            ▼            ▼
       │  (CoT + Few-Shot Tag)  │                      [Lookup]     [Analysis]   [Advisor]
       └────────────┬───────────┘                          │            │            │
                    │                                      │            │            │
                    ▼                                      ▼            │            ▼
         [ Clean Categorized DF ] ◄────────────────────────┴────────────┘      [Agentic RAG]
                                                                                     │
                                                                                     ▼
                                                                           ┌──────────────────┐
                                                                           │     ChromaDB     │
                                                                           │ (Guidelines DB)  │
                                                                           └──────────────────┘
```

---

## 🤖 The Multi-Agent Squad

1. **Parser Agent (`ParserAgent` in `finance_agent/agents/parser_agent.py`):**
   - Ingests CSV and PDF bank/UPI statements.
   - Automatically detects headers across major Indian banks (SBI, HDFC, ICICI, Paytm, PhonePe, Axis).
   - Merges debit/credit columns into unified signed amounts (`-` for debits, `+` for credits).
   - Coerces 8 different date formats and sorts transactions chronologically.

2. **Categorizer Agent (`CategorizerAgent` in `finance_agent/agents/categorizer_agent.py`):**
   - Tags each transaction into 15 standard spending categories.
   - Hybrid pipeline: High-precision merchant heuristics for instant classification, backed by **LangChain Chain-of-Thought (CoT) Few-Shot LLM** reasoning for ambiguous descriptions.

3. **Analysis Agent (`AnalysisAgent` in `finance_agent/agents/analysis_agent.py`):**
   - Uses `PandasFinanceTool` and `FinanceCalculator` to compute exact financial totals, trends, category breakdowns, and monthly balances without mathematical hallucination.
   - Connected via LangChain Expression Language (LCEL).

4. **Advisor Agent (`AdvisorAgent` in `finance_agent/agents/advisor_agent.py`):**
   - **Agentic RAG**: Dynamically evaluates user spending ratios, detects overspending red flags (e.g., dining > 10%, housing > 35%, subscription drain, low savings rate), queries ChromaDB for relevant guidelines (50/30/20 rule, emergency funds, debt snowball), and formulates tailored, quantified savings recommendations.

5. **Intent Router (`IntentRouter` in `finance_agent/utils/router.py`):**
   - Classifies user inputs into `transaction_lookup`, `spending_analysis`, `savings_advice`, or `general_faq` using an LCEL few-shot classification prompt with structured JSON parsing.

---

## 📋 Mandatory Acceptance Criteria Verification

| # | Criterion | How It Is Implemented | Code Location |
|---|---|---|---|
| **1** | **LLM Integration** | Configured with OpenRouter / OpenAI API (`openai/gpt-4o-mini`, `gpt-4o`, or local LLMs). Robust offline fallbacks. | `finance_agent/config.py`, all agents |
| **2** | **Interactive GUI** | Streamlit UI with file uploader, 4 KPI metric cards, filterable transaction table, Plotly visualizations (donut, bar, trend), and chat interface. | `app.py`, `finance_agent/ui/components.py` |
| **3** | **Prompt Engineering** | Demonstrates **Chain of Thought (CoT)**, **Few-Shot Prompting**, and **Reflection** techniques. | `finance_agent/prompts/prompts.py` |
| **4** | **LangChain Core Components** | Uses `ChatPromptTemplate`, `PromptTemplate`, and **LangChain Expression Language (LCEL)** pipelines (`prompt \| llm \| parser`). | `finance_agent/prompts/`, `categorizer_agent.py`, `router.py`, `analysis_agent.py`, `advisor_agent.py` |
| **5** | **Agentic RAG** | Advisor Agent dynamically constructs search queries based on diagnosed risk flags and retrieves matching budgeting benchmarks from ChromaDB to produce grounded advice. | `finance_agent/agents/advisor_agent.py`, `finance_agent/utils/chroma_store.py` |
| **6** | **Intent Detection & Routing** | LCEL classifier detects user intent (`transaction_lookup`, `spending_analysis`, `savings_advice`, `general_faq`) and routes to the appropriate agent. | `finance_agent/utils/router.py` |
| **7** | **Vector Database** | Persistent ChromaDB storing embeddings of budgeting guidelines, 50/30/20 rules, category limits, and debt reduction strategies. | `finance_agent/utils/chroma_store.py`, `finance_agent/data/guidelines/` |

---

## 🚀 Setup & Execution Guide

### Option A: One-Click Run (Windows)
Double-click `run.bat` in the project root folder. It will launch the Streamlit dashboard in your default web browser automatically!

### Option B: Manual Command Line
```bash
# 1. Activate Python 3.11 environment
streamlit run app.py
```
Open your browser at **http://localhost:8501**.

---

## 🔑 LLM API Configuration

The project is pre-configured in `.env` to work with OpenRouter:
```ini
OPENAI_API_KEY=sk-or-v1-your-openrouter-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini
```
*(You can also change or update the key and model directly in the Streamlit web sidebar).*

---

## 🐙 Pushing to GitHub

The repository is already initialized and committed locally on branch `main`.
To push to your remote GitHub repository:

1. Create a new repository on [GitHub](https://github.com/new) (e.g. `statement-to-insights-agent`).
2. Link the remote and push:
```bash
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git branch -M main
git push -u origin main
```
*(Note: `.env` is listed in `.gitignore` so your API key is protected and will not be leaked to GitHub).*

---

## 🧪 Testing with Included Sample Data

1. Click **"Load Included Sample Statement"** on the Streamlit dashboard (or upload `sample_data/sample_statement.csv`).
2. Watch the Parser Agent and Categorizer Agent process the 23 transactions.
3. Review the KPI metrics, Interactive Filterable Table, and Plotly Donut Chart.
4. Try asking:
   - *"How much did I spend on Food & Dining?"*
   - *"What are my top 5 largest expenses?"*
   - *"Where can I cut spending and save money?"*
   - *"Find payments to Zomato"*
5. Click **"🔍 Multi-Agent Execution Trace & RAG Context"** beneath any assistant response to observe the intent classification, routing confidence, and retrieved ChromaDB guidelines!
