"""
Part 3 — Tools & Agents
========================
Mirrors Part3.ipynb with modern LangChain agents API.
All examples use Supply Chain / FMCG context.

This is the highest-leverage section for the SC3 BA role — "Agentic Framework"
is explicitly listed in the JD. Understand every section here.

Run from repo root:
    python scripts/part3_agents.py

Optional keys in .env (sections will be skipped gracefully if not set):
    GOOGLE_API_KEY   — for Google Custom Search tool
    GOOGLE_CSE_ID    — for Google Custom Search tool
"""

import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Startup diagnostics ────────────────────────────────────────────────────────
import requests as _requests

def _check(label: str, url: str, timeout: int = 4) -> bool:
    try:
        r = _requests.get(url, timeout=timeout)
        ok = r.status_code < 500
        print(f"  {'OK  ' if ok else 'FAIL'} {label} ({r.status_code})")
        return ok
    except Exception as e:
        print(f"  FAIL {label} — {type(e).__name__}: {e}")
        return False

print("=" * 60)
print("STARTUP DIAGNOSTICS")
print("=" * 60)
_check("OpenAI API",  "https://api.openai.com/v1/models")
_check("Wikipedia",   "https://en.wikipedia.org/api/rest_v1/page/summary/Python")
_check("Google APIs", "https://www.googleapis.com/")
print(f"  {'OK  ' if os.getenv('OPENAI_API_KEY') else 'FAIL'} OPENAI_API_KEY set in .env")
print()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model=MODEL, temperature=0)


# ── Section 1: What is a Tool? ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1 — Tools: functions the LLM can decide to call")
print("=" * 60)
# A Tool = Python function + name + description.
# The LLM reads the DESCRIPTION to decide WHEN to call it.
# Key BA responsibility: writing precise tool descriptions IS a functional requirement.
# Vague description → wrong tool selection → UAT failure.

from langchain_core.tools import tool
from datetime import date

@tool
def get_current_date(text: str) -> str:
    """Returns today's date in YYYY-MM-DD format.
    Use this whenever you need to know the current date, for example
    to calculate days until a delivery deadline or determine the current fiscal week.
    """
    return date.today().strftime("%Y-%m-%d")


@tool
def get_sku_lead_time(sku: str) -> str:
    """Returns the standard replenishment lead time in days for a given SKU.
    Input must be a SKU code (e.g. TEA-GB-EarlGrey-100).
    Use this to answer questions about when to place orders or whether stock will arrive in time.
    """
    # Simulated lookup — in production this would query your ERP / data warehouse
    lead_times = {
        "TEA-GB-EarlGrey-100": 21,
        "BEV-DE-Sparkling-500ml": 14,
        "JUC-AU-OrangeJuice-2L": 35,
        "BEV-FR-StillWater-1L": 10,
    }
    days = lead_times.get(sku.strip(), 28)  # 28-day default
    return f"Lead time for {sku}: {days} days"


@tool
def calculate_reorder_point(sku: str) -> str:
    """Calculates the reorder point for a SKU.
    The reorder point is: (average daily demand × lead time) + safety stock.
    Input must be a SKU code. Returns the reorder point in units.
    Use this when a planner asks whether current stock levels require an urgent order.
    """
    # Simulated data
    data = {
        "TEA-GB-EarlGrey-100":  {"daily_demand": 120, "lead_time": 21, "safety_stock": 500},
        "BEV-DE-Sparkling-500ml": {"daily_demand": 300, "lead_time": 14, "safety_stock": 800},
    }
    if sku not in data:
        return f"No demand data available for SKU: {sku}"
    d = data[sku]
    rop = d["daily_demand"] * d["lead_time"] + d["safety_stock"]
    return (
        f"Reorder point for {sku}: {rop} units "
        f"(demand {d['daily_demand']}/day × {d['lead_time']} days lead time + "
        f"{d['safety_stock']} safety stock)"
    )


# Inspect a tool — the description IS what the LLM sees
print("Tool name:", get_current_date.name)
print("Tool description:\n", get_current_date.description)
print("\nTool args schema:", get_current_date.args)


# ── Section 2: Wikipedia tool (built-in) ──────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — Wikipedia tool (built-in)")
print("=" * 60)
# Mirrors Part3 Task 1. WikipediaQueryRun is a pre-built tool.
# Use case: agent looks up commodity background before generating a risk report.

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

print("Tool name:", wiki_tool.name)
try:
    result = wiki_tool.run("Darjeeling tea supply chain")
    print("Wikipedia result (truncated):")
    print(result[:300])
except Exception as e:
    print(f"WARNING: Wikipedia blocked (likely corporate firewall) — {type(e).__name__}: {e}")
    print("Continuing without Wikipedia. Agent sections below will use custom tools only.")
    wiki_tool = None


# ── Section 3: Build an agent with custom tools ────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — Agent: LLM + tools + ReAct reasoning loop")
print("=" * 60)
# ReAct loop: Thought → Action (pick tool) → Observation (tool output) → repeat → Answer
# The LLM reasons about WHICH tool to call and WHEN to stop.
# BA responsibility: define the tool set, the system prompt, and the stopping conditions.
# These become functional + non-functional requirements in the BRD.

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Pull the standard ReAct prompt from LangChain hub
# (defines the Thought/Action/Observation format the agent follows)
react_prompt = hub.pull("hwchase17/react")

sc_tools = [get_current_date, get_sku_lead_time, calculate_reorder_point]
if wiki_tool:
    sc_tools.append(wiki_tool)

sc_agent = AgentExecutor(
    agent=create_react_agent(llm=llm, tools=sc_tools, prompt=react_prompt),
    tools=sc_tools,
    verbose=True,               # shows Thought/Action/Observation — read this for UAT!
    handle_parsing_errors=True,
    max_iterations=6,           # NFR: prevent runaway agents — document this limit
)

print("\nAgent query 1: Lead time question")
print("-" * 40)
response = sc_agent.invoke({"input":
    "What is the lead time for TEA-GB-EarlGrey-100, and is today early enough "
    "to place an order for delivery within 30 days?"
})
print("\nFinal answer:", response["output"])


# ── Section 4: Custom tool with business logic ────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — Custom tool: reorder point agent")
print("=" * 60)

print("\nAgent query 2: Reorder calculation")
print("-" * 40)
response = sc_agent.invoke({"input":
    "What is the reorder point for BEV-DE-Sparkling-500ml? "
    "Based on today's date, when should we place the next replenishment order "
    "if we want stock arriving before March 1st?"
})
print("\nFinal answer:", response["output"])


# ── Section 5: Agent in a chain ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — Agent inside a chain (most important pattern)")
print("=" * 60)
# An agent can be a step inside a larger LCEL chain.
# Pattern from Part3 notebook: chain1 determines WHAT to research,
# then the agent does the research, then chain2 formats the output.
# This is the "agentic framework" the JD refers to.

# Step 1: LLM selects the most relevant SKU to investigate
select_prompt = ChatPromptTemplate.from_template("""
Given this business question, identify the single most relevant SKU code to investigate.
Return ONLY the SKU code, nothing else.

Available SKUs: TEA-GB-EarlGrey-100, BEV-DE-Sparkling-500ml, JUC-AU-OrangeJuice-2L

Question: {question}
""")
select_chain = select_prompt | llm | StrOutputParser()

# Step 2: agent investigates that SKU
def run_sc_agent(sku: str) -> str:
    response = sc_agent.invoke({"input":
        f"Provide a complete replenishment brief for SKU: {sku}. "
        f"Include lead time, reorder point, and any background on the product category."
    })
    return response["output"]

# Step 3: format as an executive summary
format_prompt = ChatPromptTemplate.from_template("""
Format this supply chain brief as a structured email to a Category Manager.
Include a subject line, 3 key bullet points, and a recommended action.

Brief:
{brief}
""")
format_chain = format_prompt | llm | StrOutputParser()

# Full pipeline: question → select SKU → agent research → format email
full_pipeline = (
    select_chain
    | (lambda sku: run_sc_agent(sku))
    | (lambda brief: {"brief": brief})
    | format_chain
)

print("\nFull pipeline: question → SKU selection → agent → formatted email")
print("-" * 40)
result = full_pipeline.invoke({
    "question": "We're running low on our premium sparkling water in Germany, what should we do?"
})
print(result)


# ── Section 6: Google Custom Search tool (optional) ───────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — Google Custom Search tool (optional)")
print("=" * 60)

google_key = os.getenv("GOOGLE_API_KEY", "")
google_cse = os.getenv("GOOGLE_CSE_ID", "")

if not google_key or not google_cse:
    print("GOOGLE_API_KEY or GOOGLE_CSE_ID not set — skipping.")
    print("Add both to .env to enable (see Part3 Task 2 for setup instructions).")
else:
    from googleapiclient.discovery import build

    @tool
    def google_search(query: str) -> str:
        """Search the web using Google Custom Search.
        Use this to find current news, market prices, supplier information,
        or any recent information that is not in the model's training data.
        Input should be a specific search query string.
        """
        try:
            service = build("customsearch", "v1", developerKey=google_key)
            results = service.cse().list(q=query, cx=google_cse, num=3).execute()
            snippets = [item.get("snippet", "") for item in results.get("items", [])]
            return "\n---\n".join(snippets)
        except Exception as e:
            return f"Search failed: {e}"

    web_tools = [google_search, get_current_date]
    if wiki_tool:
        web_tools.append(wiki_tool)
    web_agent = AgentExecutor(
        agent=create_react_agent(llm=llm, tools=web_tools, prompt=react_prompt),
        tools=web_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    response = web_agent.invoke(
        "What are the current global tea commodity prices and how might they affect "
        "a UK beverage manufacturer's procurement costs this quarter?"
    )
    print("Final answer:", response["output"])


# ── Section 7: Key BA takeaways on agents ─────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — BA interview talking points on agents")
print("=" * 60)

talking_points = """
1. TOOL DESCRIPTION = FUNCTIONAL REQUIREMENT
   The LLM reads the tool description to decide when to call it.
   Vague description → wrong tool call → defect in UAT.
   Write tool descriptions with the same rigour as INVEST acceptance criteria.

2. VERBOSE=TRUE IS YOUR UAT WINDOW
   The Thought/Action/Observation trace shows the agent's reasoning chain.
   In UAT, you verify: Did it pick the right tool? Did it stop at the right time?
   This trace is the agent's 'audit log' — an NFR for regulated environments.

3. MAX_ITERATIONS = SAFETY GUARDRAIL
   Unbounded agents will loop, burn tokens, and time out.
   Document max_iterations as a non-functional requirement: "Agent must return
   a result or graceful failure within N reasoning steps."

4. HANDLE_PARSING_ERRORS = RESILIENCE NFR
   Agents sometimes generate malformed tool calls. handle_parsing_errors=True
   lets the agent self-correct. Document the expected error rate in the BRD.

5. AGENT IN A CHAIN = AGENTIC WORKFLOW
   Combining deterministic chains (prompt|model|parser) with non-deterministic
   agents (tool selection) is the core pattern of agentic frameworks.
   BA job: define which steps are deterministic (testable, auditable) and
   which require agent autonomy (where UAT needs adversarial test cases).
"""
print(talking_points)

print("=" * 60)
print("Part 3 complete. You have covered the full LangChain workshop.")
print("=" * 60)
