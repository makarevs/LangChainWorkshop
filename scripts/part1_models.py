"""
Part 1 — Models, Prompts, Output Parsers
=========================================
Mirrors Part1.ipynb with modern langchain-openai imports.
All examples use Supply Chain / O2C context.

Run from repo root:
    python scripts/part1_models.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ── Section 1: LLM vs ChatModel ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1 — LLM vs ChatModel")
print("=" * 60)
# ChatModel takes a list of role-tagged messages (System / Human / AI).
# This role structure matters for SC agents: System sets persona/constraints,
# Human carries the business question, AI carries prior context if needed.

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model=MODEL, temperature=0)

# Simple single-message call
response = llm.invoke([HumanMessage(content="When did Coca-Cola launch its first product?")])
print("Single HumanMessage:", response.content)

# System + Human — more control over model persona
messages = [
    SystemMessage(content="You are a supply chain expert specialising in FMCG beverage logistics."),
    HumanMessage(content="What are the top 3 risks in cold-chain distribution for chilled beverages?"),
]
response = llm.invoke(messages)
print("\nSystem + Human:\n", response.content)


# ── Section 2: PromptTemplate ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — ChatPromptTemplate")
print("=" * 60)

from langchain_core.prompts import ChatPromptTemplate

# Basic template with a single variable
simple_template = ChatPromptTemplate.from_template(
    "Describe the Order-to-Cash process for a {company_type} company in 3 bullet points."
)
messages = simple_template.format_messages(company_type="global beverage manufacturer")
print("Formatted prompt:")
print(messages[0].content)

llm_response = llm.invoke(messages)
print("\nModel response:")
print(llm_response.content)


# ── Section 3: Multi-variable template ────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — Multi-variable template + temperature")
print("=" * 60)

translate_template = ChatPromptTemplate.from_template("""
Translate the following supply chain alert from {source_language} to {target_language}.
After translation, add a one-line {style} commentary for the receiving team.

Alert: ```{alert_text}```
""")

# Temperature 0 — deterministic translation (right call for operational alerts)
translate_llm = ChatOpenAI(model=MODEL, temperature=0)

messages = translate_template.format_messages(
    source_language="German",
    target_language="English",
    style="professional",
    alert_text="Lieferverzögerung: Palette TEA-GB-EarlGrey-100 um 48 Stunden verspätet "
               "aufgrund Zollkontrolle in Hamburg.",
)
response = translate_llm.invoke(messages)
print(response.content)


# ── Section 4: Structured output — JSON extraction ───────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — StructuredOutputParser (typed dict from model)")
print("=" * 60)
# This is how you extract structured data from unstructured documents —
# a key BA skill when defining data quality rules for ingestion pipelines.

from langchain.output_parsers import ResponseSchema, StructuredOutputParser

po_schemas = [
    ResponseSchema(name="po_number",    description="Purchase order number", type="string"),
    ResponseSchema(name="supplier",     description="Supplier name",          type="string"),
    ResponseSchema(name="sku",          description="Product SKU or code",    type="string"),
    ResponseSchema(name="quantity",     description="Ordered quantity (units)",type="integer"),
    ResponseSchema(name="delivery_date",description="Expected delivery date (YYYY-MM-DD)", type="string"),
    ResponseSchema(name="incoterm",     description="Incoterm (e.g. CIF, FOB, DAP)",       type="string"),
]

parser = StructuredOutputParser.from_response_schemas(po_schemas)
format_instructions = parser.get_format_instructions()

po_template = ChatPromptTemplate.from_template("""
Extract the purchase order details from the following email excerpt.

{format_instructions}

Email:
```{email_text}```
""")

email_text = """
Dear Procurement Team,

Please find attached PO-2026-04782 for 2,400 cases of Earl Grey Tea 100s (SKU: TEA-GB-EarlGrey-100).
Supplier: Darjeeling Fine Teas Ltd. Delivery expected by 2026-05-15.
Terms: CIF Liverpool port.

Regards,
Ramesh Patel, Supply Planning
"""

po_chain = po_template | llm | parser

result = po_chain.invoke({
    "format_instructions": format_instructions,
    "email_text": email_text,
})

print(f"Type: {type(result)}")
print(result)
print(f"\nPO: {result['po_number']}  |  SKU: {result['sku']}  |  Qty: {result['quantity']}")
print(f"Deliver by: {result['delivery_date']}  |  Incoterm: {result['incoterm']}")


# ── Section 5: DatetimeOutputParser + OutputFixingParser ──────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — DatetimeOutputParser + self-healing OutputFixingParser")
print("=" * 60)
# OutputFixingParser: if the model returns a malformed date, a second LLM call
# auto-corrects it. Use this as a pattern for NFR: resilience / self-healing.

from langchain.output_parsers import DatetimeOutputParser, OutputFixingParser

datetime_parser = DatetimeOutputParser()

date_template = ChatPromptTemplate.from_template("""
{question}
{format_instructions}
""")

# First: without fixing parser — may fail if model format drifts
date_chain = date_template | llm | datetime_parser

try:
    result = date_chain.invoke({
        "question": "When was the World Trade Organization (WTO) founded?",
        "format_instructions": datetime_parser.get_format_instructions(),
    })
    print(f"Date (direct parser): {result}")
except Exception as e:
    print(f"Direct parser failed: {e}")
    # Self-healing fallback
    fix_parser = OutputFixingParser.from_llm(parser=datetime_parser, llm=llm)
    fix_chain = date_template | llm | fix_parser
    result = fix_chain.invoke({
        "question": "When was the World Trade Organization (WTO) founded?",
        "format_instructions": datetime_parser.get_format_instructions(),
    })
    print(f"Date (fix parser): {result}")


# ── Section 6: HuggingFace model (optional — needs HUGGINGFACEHUB_API_TOKEN) ──
print("\n" + "=" * 60)
print("SECTION 6 — HuggingFace model (skip if no token set)")
print("=" * 60)

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
if not hf_token or hf_token == "hf_...":
    print("HUGGINGFACEHUB_API_TOKEN not set — skipping HuggingFace section.")
    print("Add token to .env to enable. Workshop Part 1 Task 5 uses this.")
else:
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain_core.output_parsers import StrOutputParser

    hf_llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        task="text2text-generation",
        max_new_tokens=100,
    )
    hf_chain = ChatPromptTemplate.from_template("{question}") | hf_llm | StrOutputParser()
    result = hf_chain.invoke({"question": "What does SKU stand for in supply chain?"})
    print(f"HuggingFace (flan-t5-base): {result}")


print("\n" + "=" * 60)
print("Part 1 complete. Next: run scripts/part2_chains.py")
print("=" * 60)
