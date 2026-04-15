"""
Part 2 — Chains (LCEL)
=======================
Mirrors Part2.ipynb with modern LCEL syntax (no deprecated LLMChain / SequentialChain).
All examples use Supply Chain / FMCG context.

Run from repo root:
    python scripts/part2_chains.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model=MODEL, temperature=0.3)


# ── Section 1: Basic chain — prompt | model | parser ──────────────────────────
print("\n" + "=" * 60)
print("SECTION 1 — Basic LCEL chain")
print("=" * 60)
# The | operator connects Runnables. Every chain has .invoke() / .batch() / .stream()
# This is the pattern you describe in BRD as "single-step AI transformation".

basic_prompt = ChatPromptTemplate.from_template(
    "Describe the role of a {role} in an FMCG supply chain in 3 bullet points."
)
basic_chain = basic_prompt | llm | StrOutputParser()

result = basic_chain.invoke({"role": "Demand Planner"})
print(result)


# ── Section 2: Streaming ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — Streaming (token-by-token)")
print("=" * 60)
# NFR: a planning assistant UI should stream to avoid blank-screen wait.
# .stream() is drop-in — same chain, different call method.

stream_prompt = ChatPromptTemplate.from_template(
    "Write a briefing note on the supply chain risks of sourcing {ingredient} "
    "from {origin} for a UK-based beverage manufacturer."
)
stream_chain = stream_prompt | llm | StrOutputParser()

print("Streaming:")
print("-" * 40)
for chunk in stream_chain.stream({"ingredient": "Darjeeling tea leaves", "origin": "India"}):
    print(chunk, end="", flush=True)
print("\n" + "-" * 40)


# ── Section 3: Batch processing ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — Batch processing")
print("=" * 60)
# .batch() runs the chain over a list in parallel — essential for S&OP bulk runs.
# max_concurrency prevents hitting OpenAI rate limits.

batch_prompt = ChatPromptTemplate.from_template(
    "In one sentence, state the biggest O2C risk for a {market} market."
)
batch_chain = batch_prompt | llm | StrOutputParser()

markets = [
    {"market": "United Kingdom"},
    {"market": "Germany"},
    {"market": "India"},
    {"market": "Brazil"},
]

results = batch_chain.batch(markets, config={"max_concurrency": 4})
for market, result in zip(markets, results):
    print(f"{market['market']:>20}: {result}")


# ── Section 4: Sequential chain (two chained LLM calls) ───────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — Sequential chain: two chained LLM calls")
print("=" * 60)
# Modern replacement for deprecated SequentialChain.
# Pattern: chain1 | (lambda x: {"key": x}) | chain2
# Interview frame: "classify anomaly → auto-draft remediation ticket"

# Step 1: generate a supply chain incident report
incident_prompt = ChatPromptTemplate.from_template(
    "Write a concise supply chain incident report for: {incident_description}"
)
incident_chain = incident_prompt | llm | StrOutputParser()

# Step 2: extract key action items from the report
actions_prompt = ChatPromptTemplate.from_template("""
You are a Business Analyst. Extract the top 3 action items from this incident report
and format them as Azure DevOps tasks with Priority and Assignee role.

Report:
{report}
""")
actions_chain = actions_prompt | llm | StrOutputParser()

# Connect chains — output of step 1 → input of step 2
full_chain = incident_chain | (lambda report: {"report": report}) | actions_chain

result = full_chain.invoke({
    "incident_description": (
        "Distribution centre in Hamburg ran out of buffer stock for TEA-GB-EarlGrey-100 "
        "during peak December demand due to a missed replenishment order."
    )
})
print(result)


# ── Section 5: RunnableParallel — two branches, one input ─────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — RunnableParallel: fan-out to multiple chains")
print("=" * 60)
# Run two independent LLM calls simultaneously and merge the outputs.
# Use case: enrich a SKU with both commercial and logistics context in one call.

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

commercial_prompt = ChatPromptTemplate.from_template(
    "Describe the commercial strategy for launching {sku} in {market} in 2 sentences."
)
logistics_prompt = ChatPromptTemplate.from_template(
    "Describe the logistics challenges for distributing {sku} in {market} in 2 sentences."
)

parallel_chain = RunnableParallel(
    commercial=commercial_prompt | llm | StrOutputParser(),
    logistics=logistics_prompt | llm | StrOutputParser(),
    sku=RunnablePassthrough() | (lambda x: x["sku"]),   # pass-through for context
)

result = parallel_chain.invoke({
    "sku": "Premium Iced Tea 500ml",
    "market": "India",
})

print(f"SKU: {result['sku']}")
print(f"\nCommercial:\n{result['commercial']}")
print(f"\nLogistics:\n{result['logistics']}")


# ── Section 6: Three-step pipeline — the workshop's "multiple chains" task ─────
print("\n" + "=" * 60)
print("SECTION 6 — Three-step pipeline: describe → review → summarise")
print("=" * 60)
# Mirrors Part2 Task 3 (restaurant dish) but with SC content.
# Also mirrors the HuggingFace summarisation step from Part2 notebook.

# Step 1: describe a new product concept
describe_prompt = ChatPromptTemplate.from_template(
    "Write a 2-paragraph product concept description for: {product_concept}"
)
describe_chain = describe_prompt | llm | StrOutputParser()

# Step 2: review it as a supply chain director
review_prompt = ChatPromptTemplate.from_template("""
You are a Supply Chain Director at a global beverage company.
Review this product concept from a manufacturability and distribution perspective.
Be critical and specific.

Concept:
{description}
""")
review_chain = review_prompt | llm | StrOutputParser()

# Step 3: condense the review to a one-line executive decision
decision_prompt = ChatPromptTemplate.from_template(
    "Summarise this supply chain review in exactly one sentence, "
    "starting with GO or NO-GO:\n\n{review}"
)
decision_chain = decision_prompt | llm | StrOutputParser()

# Full three-step pipeline
pipeline = (
    describe_chain
    | (lambda d: {"description": d})
    | review_chain
    | (lambda r: {"review": r})
    | decision_chain
)

result = pipeline.invoke({
    "product_concept": "A line of cold-brew tea cans in 250ml, 500ml, and 1L formats "
                       "targeting gym-goers in the UK, sourced from Sri Lanka."
})
print(f"Executive decision: {result}")


# ── Section 7: OutputFixingParser in a chain ───────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — OutputFixingParser: self-healing chain")
print("=" * 60)
# When a model occasionally returns malformed structured output,
# OutputFixingParser sends it to a second LLM call that corrects the format.
# Key NFR: resilience — the system auto-recovers without human intervention.

from langchain.output_parsers import ResponseSchema, StructuredOutputParser, OutputFixingParser

schema = [
    ResponseSchema(name="decision",   description="GO or NO-GO", type="string"),
    ResponseSchema(name="rationale",  description="One sentence rationale", type="string"),
    ResponseSchema(name="risk_score", description="Risk score 1-10 (10 = highest risk)", type="integer"),
]

struct_parser = StructuredOutputParser.from_response_schemas(schema)
fix_parser = OutputFixingParser.from_llm(parser=struct_parser, llm=llm)

decision_struct_prompt = ChatPromptTemplate.from_template("""
You are a Supply Chain Director. Evaluate this new product concept.

Concept: {product_concept}

{format_instructions}
""")

struct_chain = decision_struct_prompt | llm | fix_parser

result = struct_chain.invoke({
    "product_concept": "Aluminium-canned sparkling water with electrolytes, targeting "
                       "sports events in Germany, manufactured in Poland.",
    "format_instructions": struct_parser.get_format_instructions(),
})

print(f"Type: {type(result)}")
print(f"Decision:   {result['decision']}")
print(f"Rationale:  {result['rationale']}")
print(f"Risk score: {result['risk_score']}/10")


print("\n" + "=" * 60)
print("Part 2 complete. Next: run scripts/part3_agents.py")
print("=" * 60)
