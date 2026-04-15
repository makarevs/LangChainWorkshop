"""
Part 0 — LangChain Warmup
=========================
Run each section independently by setting a breakpoint or commenting out sections below.
All examples use Supply Chain / O2C scenarios for a global beverage FMCG company.

Run from repo root:
    python scripts/part0_warmup.py
"""

import os
from dotenv import load_dotenv

# ── Section 1: Load API key ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1 — Load API key from .env")
print("=" * 60)

load_dotenv()

key = os.getenv("OPENAI_API_KEY", "NOT SET")
print(f"Key loaded: {key[:8]}..." if key != "NOT SET" else "ERROR: OPENAI_API_KEY not set in .env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
print(f"Using model: {MODEL}")


# ── Section 2: ChatModel ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — ChatModel: the LLM wrapper")
print("=" * 60)
# temperature=0  → deterministic  → use for data extraction, classification
# temperature=0.7 → creative      → use for drafting, brainstorming

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=MODEL, temperature=0)

response = llm.invoke("In one sentence: what is Order-to-Cash (O2C)?")
print(response.content)


# ── Section 3: PromptTemplate ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — PromptTemplate: parameterised prompts")
print("=" * 60)
# The template IS the functional specification — version-control it like code.

from langchain_core.prompts import ChatPromptTemplate

anomaly_template = ChatPromptTemplate.from_template("""
You are a supply chain analyst at a global beverage manufacturer.
Classify the following demand forecast anomaly and suggest the most likely root cause.

SKU: {sku}
Market: {market}
Anomaly: {anomaly_description}

Respond in exactly this format:
Classification: <one of: Demand Spike, Demand Drop, Data Error, Seasonal Shift>
Likely cause: <one sentence>
Recommended action: <one sentence>
""")

# format_messages() builds the prompt — no LLM call yet
messages = anomaly_template.format_messages(
    sku="TEA-GB-EarlGrey-100",
    market="United Kingdom",
    anomaly_description="Forecast shows +340% spike in week 48 vs 4-week average",
)
print("--- Prompt that will be sent to the model ---")
print(messages[0].content)


# ── Section 4: LCEL pipe operator ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — LCEL: prompt | model | parser")
print("=" * 60)
# Each step is independently testable — mock the model to test just the template.

from langchain_core.output_parsers import StrOutputParser

anomaly_chain = anomaly_template | llm | StrOutputParser()

result = anomaly_chain.invoke({
    "sku": "BEV-DE-Sparkling-500ml",
    "market": "Germany",
    "anomaly_description": "Forecast shows -60% drop starting week 2 of January",
})
print(result)


# ── Section 5: batch() ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — batch(): process multiple SKUs in parallel")
print("=" * 60)
# NFR: system must classify all weekly S&OP anomalies within 5 minutes.
# max_concurrency controls parallel API calls — tune to avoid rate limits.

weekly_anomalies = [
    {
        "sku": "TEA-GB-EarlGrey-100",
        "market": "United Kingdom",
        "anomaly_description": "+340% spike in week 48 — Black Friday promotional period",
    },
    {
        "sku": "BEV-FR-StillWater-1L",
        "market": "France",
        "anomaly_description": "Negative forecast value (-200 cases) in week 3",
    },
    {
        "sku": "JUC-AU-OrangeJuice-2L",
        "market": "Australia",
        "anomaly_description": "-45% drop aligned with competitor launching lower-price SKU",
    },
]

results = anomaly_chain.batch(weekly_anomalies, config={"max_concurrency": 3})

for anomaly, result in zip(weekly_anomalies, results):
    print(f"\n=== {anomaly['sku']} ({anomaly['market']}) ===")
    print(result)


# ── Section 6: StructuredOutputParser ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — StructuredOutputParser: typed dict instead of string")
print("=" * 60)
# The parser schema IS the acceptance criterion — if parse() fails, the feature fails.

from langchain.output_parsers import ResponseSchema, StructuredOutputParser

response_schemas = [
    ResponseSchema(name="classification",
                   description="One of: Demand Spike, Demand Drop, Data Error, Seasonal Shift",
                   type="string"),
    ResponseSchema(name="root_cause",
                   description="Most likely business reason for the anomaly",
                   type="string"),
    ResponseSchema(name="action",
                   description="Recommended action for the supply planner",
                   type="string"),
    ResponseSchema(name="confidence",
                   description="Confidence score 0-10 for this classification",
                   type="integer"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

structured_template = ChatPromptTemplate.from_template("""
You are a supply chain analyst at a global beverage manufacturer.
Classify this demand forecast anomaly.

SKU: {sku}
Market: {market}
Anomaly: {anomaly_description}

{format_instructions}
""")

structured_chain = structured_template | llm | output_parser

result = structured_chain.invoke({
    "sku": "TEA-GB-EarlGrey-100",
    "market": "United Kingdom",
    "anomaly_description": "+340% spike in week 48 — Black Friday promotional period",
    "format_instructions": format_instructions,
})

print(f"Type returned: {type(result)}")  # dict — not string!
print(result)
print(f"\nConfidence: {result['confidence']}/10")
print(f"Action: {result['action']}")


# ── Section 7: Multi-step chain ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — Multi-step chain: classify → draft AzDO user story")
print("=" * 60)

classify_template = ChatPromptTemplate.from_template("""
Classify this supply chain anomaly in one sentence.
SKU: {sku}, Market: {market}
Anomaly: {anomaly_description}
""")
classify_chain = classify_template | llm | StrOutputParser()

azdo_template = ChatPromptTemplate.from_template("""
You are a Business Analyst writing Azure DevOps work items.
Based on this supply chain anomaly classification:

{classification}

Write a User Story in INVEST format:
- Title: As a [role], I want [goal] so that [benefit]
- Acceptance Criteria (3 bullet points)
- Definition of Done (2 bullet points)
""")
azdo_chain = azdo_template | llm | StrOutputParser()

# Connect: output of classify_chain feeds into azdo_chain
full_pipeline = classify_chain | (lambda c: {"classification": c}) | azdo_chain

story = full_pipeline.invoke({
    "sku": "TEA-GB-EarlGrey-100",
    "market": "United Kingdom",
    "anomaly_description": "+340% spike in week 48 — Black Friday promotional period",
})
print(story)


# ── Section 8: stream() ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 8 — stream(): token-by-token output")
print("=" * 60)
# NFR: for a planning assistant UI, streaming prevents a blank-screen wait.
# Document streaming as a non-functional requirement in the BRD.

stream_template = ChatPromptTemplate.from_template("""
Write a 3-paragraph executive summary of the supply chain risks
for a global beverage manufacturer entering the {market} market
with a new {product_type} product line.
""")
stream_chain = stream_template | llm | StrOutputParser()

print("Streaming response:")
print("-" * 40)
for chunk in stream_chain.stream({"market": "India", "product_type": "premium iced tea"}):
    print(chunk, end="", flush=True)
print("\n" + "-" * 40)


# ── Section 9: Temperature experiment ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 9 — Temperature: determinism vs creativity")
print("=" * 60)
# 0.0 → data extraction, classification, SQL generation
# 0.3 → drafting work items, summaries
# 0.7 → creative content, brainstorming
# 1.0+ → rarely useful in enterprise SC contexts

temp_template = ChatPromptTemplate.from_template(
    "Suggest a catchy product name for a new premium tea blend "
    "targeting health-conscious consumers in {market}."
)

for temp in [0.0, 0.7, 1.2]:
    creative_llm = ChatOpenAI(model=MODEL, temperature=temp)
    chain = temp_template | creative_llm | StrOutputParser()
    result = chain.invoke({"market": "United Kingdom"})
    print(f"Temperature {temp:>4}: {result}")


print("\n" + "=" * 60)
print("Part 0 complete. Next: run scripts/part1_models.py")
print("=" * 60)
