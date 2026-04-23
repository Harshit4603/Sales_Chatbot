import os
import time
import json
import asyncio
import requests
from dotenv import load_dotenv
import re

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from groq import Groq
from pinecone import Pinecone
import google.genai as genai
from google.genai import types

from database import get_db
from models import ChatSession, ChatMessage, Employee, get_ist
from fastapi import File, UploadFile
import tempfile
import shutil

# import ingest functions
from ingest import (
    extract_docx, extract_pdf, extract_pptx,
    chunk_all_sections, embed_all, upload_to_pinecone,
    infer_doc_category
)

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

SOURCES_ACCESSED    = 10     # top_k chunks pulled from Pinecone per query
MEMORY_TURNS        = 5      # how many past Q&A pairs to include in the LLM prompt
DB_STRONG_THRESHOLD = 4      # min strong DB chunks to consider DB context "rich"
SCORE_THRESHOLD     = 0.25   # min Pinecone score to count a chunk as "strong"
MAX_CONTEXT_CHARS   = 4500   # cap on DB context fed to LLM

# =============================================================================
# CLIENTS
# =============================================================================

HF_API_KEY  = os.getenv("HF_API_KEY")
HF_API_URL  = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5"
HF_HEADERS  = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = "sales-chatbot"
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

print("HF_API_KEY    :", "✅" if HF_API_KEY    else "❌ MISSING")
print("PINECONE_KEY  :", "✅" if PINECONE_API_KEY else "❌ MISSING")
print("GROQ_API_KEY  :", "✅" if GROQ_API_KEY   else "❌ MISSING")
print("GEMINI_API_KEY:", "✅" if GEMINI_API_KEY  else "❌ MISSING")

pc           = Pinecone(api_key=PINECONE_API_KEY)
index        = pc.Index(PINECONE_INDEX)
groq_client  = Groq(api_key=GROQ_API_KEY)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatRequest(BaseModel):
    employee_id: str
    session_id:  str | None = None
    query:       str
    role:        str | None = None   # "sales" | "employee"

class RatingRequest(BaseModel):
    rating: str

class LoginRequest(BaseModel):
    email:    str
    password: str

# =============================================================================
# ROUTING SIGNALS
# Centralised keyword lists used by the router. Edit here, not scattered
# throughout the codebase.
# =============================================================================

# These trigger "retrieval" even when classifier says otherwise
PRODUCT_SIGNALS = {
    "sofa", "mattress", "pillow", "recliner", "bed", "chair",
    "recommend", "suggest", "option", "catalog", "price", "warranty",
    "sop", "policy", "leave", "training", "onboarding",
    "dimension", "size", "color", "colour", "shade", "variant",
    "available in", "how many", "best", "which",
}

# These force the COMPARISON sub-route (Gemini-first)
COMPARISON_SIGNALS = {
    "vs", "versus", "better than", "compare", "comparison",
    "which one", "which is better", "difference between",
    "wakefit", "sleepwell", "duroflex", "sunday", "competitor",
    "against",
}

# doc_categories that are "internal-only" — DB is authoritative
INTERNAL_CATEGORIES = {"policy", "sop", "training"}

# Role-based Pinecone filter
ROLE_CATEGORY_ALLOW = {
    "sales":    {"product", "pricing", "faq", "general"},
    "employee": {"product", "policy", "sop", "training", "pricing", "faq", "general"},
}

def detect_and_translate(user_query: str) -> dict:
    """Detects language and translates to English if non-English/non-Hinglish."""
    prompt = f"""Detect the language of this query and translate to English if needed.

Query: {user_query}

Rules:
- If English → return as is
- If Hinglish (Hindi+English mix) → return as is, it is supported natively
- If any other language (Marathi, Tamil, Telugu, Gujarati, Kannada, Bengali etc.) → translate to English
- Always identify the original language

Return ONLY valid JSON:
{{
  "original_language": "english|hinglish|hindi|marathi|tamil|telugu|gujarati|kannada|bengali|other",
  "translated_query": "<translated to English, or original if english/hinglish>",
  "needs_translation": true | false
}}"""

    try:
        resp = groq_client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=150,
)
        raw = resp.choices[0].message.content.strip()
        print(f"[Translator] Raw response: '{raw[:300]}'")

        # Strip thinking blocks if present
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        # Extract JSON
        raw = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        raw = match.group() if match else raw
        result = json.loads(raw)
        print(f"[Translator] lang={result.get('original_language')} | "
              f"needs_translation={result.get('needs_translation')}")
        return {
            "original_language": result.get("original_language", "english"),
            "translated_query":  result.get("translated_query", user_query),
            "needs_translation": bool(result.get("needs_translation", False))
        }
    except Exception as e:
        print(f"[Translator] Failed ({e}) — using original query")
        return {
            "original_language": "english",
            "translated_query":  user_query,
            "needs_translation": False
        }

# =============================================================================
# STEP 1 — QUERY PARSER
# =============================================================================

def parse_query(user_query: str) -> dict:
    prompt = f"""You are a query router for 'The Sleep Company' internal sales assistant.
The Sleep Company sells: sofas, mattresses, pillows, recliners, beds, bed frames.
Employees are sales reps and staff who help customers and manage operations.

Route the query to the correct handler based on descriptions below.

═══════════════════════════════════════════════════════
HANDLER DESCRIPTIONS
═══════════════════════════════════════════════════════

GIBBERISH HANDLER:
Receives: Random characters, incoherent text, empty noise
Examples: "asdfgh", "123abc!!!", "blue the if running potato"
→ Set gibberish=true

CONVERSATIONAL HANDLER:
Receives: Pure greetings, small talk, thanks, bye — no information needed
System: Responds warmly, no data retrieval happens at all
Examples: "Hi!", "Good morning", "Thanks", "You're helpful", "Bye"
→ Set conversation_type=chit_chat

INTERNAL HANDLER:
Receives: Queries answerable from stable company documents
System: Searches internal Pinecone knowledge base → Groq generates answer
Handles:
  - Product specs, dimensions, materials, technology, warranty
  - SOPs, return process, escalation, complaint handling
  - HR policies, leave, attendance, conduct, onboarding, training
  - Product catalog, configurations, series, variants
  - Recommendations from company offerings for a need/use case
Examples: "Valencia sofa dimensions?", "What is SmartGRID?",
          "Return policy for damaged sofa?", "Best mattress for back pain?"
→ Set needs_internal_docs=true, needs_live_data=false

LIVE HANDLER:
Receives: Queries needing current real-time data from the web
System: Gemini performs live Google Search — internal DB excluded
Handles:
  - Current pricing, EMI, discounts, festive offers
  - Current colors, variants, shades, stock availability
  - Competitor info, market data, brand comparisons
  - Anything that changes week-to-week
Examples: "Price of Valencia sofa?", "Colors available in SmartGRID?",
          "Any ongoing offers?", "How does Wakefit compare in price?"
→ Set needs_internal_docs=false, needs_live_data=true

SALES ASSIST HANDLER:
Receives: Queries needing BOTH stable internal knowledge AND live web data
System: Pinecone DB + Gemini web search both run, results merged
Handles:
  - Comparisons between two products or brands
  - Open-ended product help needing specs + live info
  - Budget-based recommendations needing catalog + pricing
  - Competitor objection handling needing internal specs + competitor data
  - Any query where internal knowledge alone OR live search alone is insufficient
Examples: "Valencia vs Luxe sofa", "Help me with Valencia sofa",
          "Best sofa under 50k", "Customer says Wakefit is cheaper than our pillow"
→ Set needs_internal_docs=true, needs_live_data=true

═══════════════════════════════════════════════════════
ROUTING RULES (apply in order):
═══════════════════════════════════════════════════════
1. Incoherent/random → gibberish
2. Pure greeting/small talk → chit_chat
3. Needs both stable knowledge + live data → sales_assist
4. Needs only stable company knowledge → internal
5. Needs only current/live data → live
6. Uncertain → sales_assist (safest default, runs everything)

═══════════════════════════════════════════════════════
OUTPUT — return ONLY this JSON, no explanation:
═══════════════════════════════════════════════════════
{{
  "gibberish":           true | false,
  "conversation_type":   "chit_chat" | "work_query",
  "needs_internal_docs": true | false,
  "needs_live_data":     true | false,
  "topic": "<max 6 words, empty string if chit_chat or gibberish>"
}}

User query: {user_query}"""
    try:
        resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"}
    )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        raw = match.group() if match else raw
        parsed = json.loads(raw)

        gibberish      = bool(parsed.get("gibberish", False))
        conv_type      = parsed.get("conversation_type", "work_query")
        needs_live     = bool(parsed.get("needs_live_data", False))
        needs_internal = bool(parsed.get("needs_internal_docs", True))
        topic          = parsed.get("topic", "")

        if not isinstance(topic, str):
            topic = ""
        if conv_type not in {"chit_chat", "work_query"}:
            conv_type = "work_query"

        # ── Routing logic ─────────────────────────────────────────────────────
        if gibberish:
            query_type, doc_category = "gibberish", "none"

        elif conv_type == "chit_chat":
            query_type, doc_category = "conversational", "none"

        else:  # work_query
            query_type = "retrieval"
            if needs_live and needs_internal:
                doc_category = "sales_assist"
            elif needs_live:
                doc_category = "live"
            elif needs_internal:
                doc_category = "internal"
            else:
                # Both false — LLM unsure, run everything
                doc_category = "sales_assist"
                needs_live = True
                needs_internal = True

        print(f"[Parser] gibberish={gibberish} | conv={conv_type} | "
              f"live={needs_live} | internal={needs_internal} | "
              f"→ {query_type}/{doc_category}")

        return {
            "query_type":     query_type,
            "doc_category":   doc_category,
            "topic":          topic,
            "needs_live":     needs_live,
            "needs_internal": needs_internal,
        }

    except Exception as e:
        print(f"[Parser] Failed ({e}) — defaulting to retrieval/internal")
        return {
            "query_type":     query_type,
            "doc_category":   doc_category,
            "topic":          topic,
            "needs_live":     needs_live,
            "needs_internal": needs_internal,
        }


# =============================================================================
# STEP 2 — QUERY REWRITER
# =============================================================================

def rewrite_query(user_query: str, memory_block: str, parsed: dict) -> str:
    """Resolves pronouns and rewrites query based on route type."""
    
    query_type   = parsed.get("query_type")
    doc_category = parsed.get("doc_category")
    needs_live   = parsed.get("needs_live")
    needs_internal = parsed.get("needs_internal")

    # ── Branch 1: Conversational — no rewrite needed ─────────────────────────
    if query_type in ("conversational", "informational"):
        return user_query

    # ── Branch 2: Live only (price/stock/competitor) ──────────────────────────
    if needs_live and not needs_internal:
        if not memory_block:
            return user_query
        prompt = f"""Rewrite as a web search query for The Sleep Company's website.
Resolve pronouns using history. Be specific about product name and what is being asked.
Return ONLY the rewritten query.

History: {memory_block}
Query: {user_query}
Rewritten:"""

    # ── Branch 3: Internal only (SOPs, policy, stable specs) ─────────────────
    elif needs_internal and not needs_live:
        if not memory_block:
            return user_query
        prompt = f"""Rewrite as a precise internal document search query.
Resolve pronouns using history. Include product/policy name and specific attribute.
Return ONLY the rewritten query.

History: {memory_block}
Query: {user_query}
Rewritten:"""

    # ── Branch 4: Both live + internal (comparison/mixed) ────────────────────
    else:
        if not memory_block:
            return user_query
        prompt = f"""Rewrite as a fully self-contained query covering both
product specs and live details (price/availability).
Resolve pronouns using history. Be explicit about both products if comparing.
Return ONLY the rewritten query.

History: {memory_block}
Query: {user_query}
Rewritten:"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        rewritten = resp.choices[0].message.content.strip()
        rewritten = rewritten.replace("```", "").strip()
        if rewritten and rewritten != user_query:
            print(f"[Rewriter] '{user_query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[Rewriter] Failed ({e}) — using original query")
        return user_query


# =============================================================================
# STEP 3 — CONVERSATIONAL HANDLER
# =============================================================================

def handle_conversational(user_query: str, memory_block: str = "") -> str:
    if not memory_block:
        prompt = f"""You are a friendly assistant for 'The Sleep Company'.
Respond warmly and naturally to the greeting in 1-2 sentences.
Mention you are there for assistance. Do NOT mention products, SOPs, or policies unless the user asks.
Just greet back and offer to help.
User message: {user_query}"""
    else:
        prompt = f"""You are a helpful assistant for 'The Sleep Company'.

Recent conversation:
{memory_block}

User just said: "{user_query}"

Greet the user warmly. If previous conversation touched on products or
policies, offer to continue. Keep it concise (3-5 sentences), professional.
Never invent prices or specs. Respond directly:"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=180,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Conversational] Error: {e}")
        return "Good morning! 👋 How can I assist you today?"


# =============================================================================
# STEP 4 — EMBEDDING
# =============================================================================

def build_query_embed_text(query: str, doc_category: str, topic: str) -> str:
    heading_hint = topic if topic else "user query"
    return f"Category: {doc_category}. Heading: {heading_hint}.\n{query}"


def get_embedding(text: str) -> list[float]:
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    result   = response.json()

    if isinstance(result, dict):
        if "error"          in result: raise Exception(f"HF API Error: {result['error']}")
        if "estimated_time" in result: raise Exception("model_loading")

    if isinstance(result, list):
        if isinstance(result[0], float):        return result
        if isinstance(result[0], list):
            if isinstance(result[0][0], float): return result[0]
            if isinstance(result[0][0], list):  return result[0][0]

    raise ValueError(f"Unexpected HF response: {str(result)[:200]}")


def get_embedding_with_retry(text: str, retries: int = 5) -> list[float]:
    wait = 5
    for attempt in range(1, retries + 1):
        try:
            return get_embedding(text)
        except Exception as e:
            msg = str(e).lower()
            if "model_loading" in msg or "503" in msg or "loading" in msg:
                print(f"  ⏳ HF model loading — retrying in {wait}s ({attempt}/{retries})")
                time.sleep(wait)
                wait = min(wait * 2, 60)
            else:
                raise
    raise Exception(f"Embedding failed after {retries} retries")


# =============================================================================
# STEP 5 — PINECONE RETRIEVAL
# =============================================================================

def generate_hypothetical_answer(user_query: str, doc_category: str) -> str:
    """Generates a hypothetical answer to improve Pinecone embedding match."""
    prompt = f"""Generate a short hypothetical answer (3-5 sentences) that would 
appear in a Sleep Company internal document for this query.
Write as if you are the document, not answering the user.
Use formal, document-like language with specific details.

Query: {user_query}
Document type: {doc_category}
Hypothetical document excerpt:"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        hypothetical = resp.choices[0].message.content.strip()
        print(f"[HyDE] Generated: {hypothetical[:80]}...")
        return hypothetical
    except Exception as e:
        print(f"[HyDE] Failed ({e}) — using original query")
        return user_query

def retrieve_from_db(
    user_query:   str,
    doc_category: str,
    topic:        str,
    role:         str | None = None,
    top_k:        int = SOURCES_ACCESSED,
) -> list[dict]:

    # Use product embedding context for sales_assist (best semantic match)
    embed_category = "product" if doc_category in ("sales_assist", "live") else doc_category
    if not embed_category or embed_category == "none":
        embed_category = "general"

    embed_text = build_query_embed_text(user_query, embed_category, topic)

    try:
        embedding = get_embedding_with_retry(embed_text)
    except Exception as e:
        print(f"[Retrieval] Embedding failed: {e}")
        return []
    if doc_category in ("internal", "sales_assist"):
        hyde_text  = generate_hypothetical_answer(user_query, doc_category)
        embed_text = build_query_embed_text(hyde_text, embed_category, topic)
    else:
        embed_text = build_query_embed_text(user_query, embed_category, topic)
        embedding      = get_embedding_with_retry(embed_text)

    allowed_categories = ROLE_CATEGORY_ALLOW.get(role) if role else None
    pinecone_filter    = {}

    # ── Build Pinecone filter ─────────────────────────────────────────────────
    if doc_category == "sales_assist":
        # Needs both product + internal docs
        allowed = ["product", "pricing", "faq", "policy", "sop", "training"]
        if role == "sales":
            allowed = ["product", "pricing", "faq"]
        pinecone_filter = {"doc_category": {"$in": allowed}}

    elif doc_category == "internal":
        allowed = ["product", "policy", "sop", "training", "faq"]
        if role == "sales":
            allowed = ["product", "faq"]
        pinecone_filter = {"doc_category": {"$in": allowed}}

    elif doc_category == "live":
        # Live queries don't hit DB meaningfully
        # but if they do fallback, pull product chunks
        pinecone_filter = {"doc_category": {"$eq": "product"}}

    elif allowed_categories:
        pinecone_filter = {"doc_category": {"$in": list(allowed_categories)}}

    else:
        pinecone_filter = {}

    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter if pinecone_filter else None,
    )
    matches = results.get("matches", [])

    # Fallback: if filter too narrow, retry without filter
    if len(matches) < 2 and pinecone_filter:
        print(f"[Retrieval] Filter too narrow — falling back to unfiltered")
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        matches = results.get("matches", [])

    valid = []
    for m in matches:
        if m["metadata"].get("text"):
            chunk           = dict(m["metadata"])
            chunk["_score"] = m.get("score", 0.0)
            valid.append(chunk)

    strong = [c for c in valid if c["_score"] >= SCORE_THRESHOLD]
    print(f"[Retrieval] {len(valid)} chunks total | {len(strong)} strong "
          f"(filter={pinecone_filter or 'none'})")
    return valid


# =============================================================================
# STEP 6 — CONTEXT BUILDER
# =============================================================================

def build_context(db_chunks: list[dict]) -> str:
    parts       = []
    total_chars = 0

    for chunk in db_chunks:
        source   = chunk.get("source", "internal document")
        heading  = chunk.get("heading", "")
        category = chunk.get("doc_category", "")
        text     = chunk.get("text", "")
        label    = f"[{category.upper()} — {source}] {heading}"
        entry    = f"{label}\n{text}"

        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            print(f"[Context] Budget reached at {len(parts)} chunks")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n\n---\n\n".join(parts)


# =============================================================================
# STEP 7 — CONVERSATION MEMORY
# =============================================================================

def build_memory_block(session_id: str, db: Session) -> str:
    if not session_id:
        return ""

    past = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(MEMORY_TURNS)
        .all()
    )
    if not past:
        return ""

    past  = list(reversed(past))
    lines = []
    for msg in past:
        lines.append(f"Previous Question: {msg.query}")
        lines.append(f"Previous Answer: {msg.answer}")

    print(f"[Memory] {len(past)} turn(s) loaded")
    return "\n".join(lines)


# =============================================================================
# STEP 8 — GROQ LLM (internal-doc-grounded answers)
# =============================================================================

GROQ_SYSTEM_PROMPT = """You are an internal assistant for sales representatives
and employees of 'The Sleep Company'.

Your role:
- Help with product recommendations
- Answer SOP, policy, and training-related questions
- Assist sales reps in customer conversations

CRITICAL RULES:
- Use ONLY the provided context
- Do NOT hallucinate product names, prices, or specs
- If you have partial information, present what you know directly without disclaimers.
- Only if you have NO information at all, say: 'I don't have that detail in our internal docs.'
- Never prefix your answer with what you don't know. Lead with what you do know.'

STYLE: Professional, Brief, concise, and point-wise. Plain text only. Answer specifically and completely. Use bullet points for lists.
Cover all relevant details — colors, sizes, features, steps — don't truncate.
No unnecessary preamble or filler sentences.
COMPANY RULE: Never portray the company negatively.

SALES MINDSET:
- Always frame answers with a positive bias toward The Sleep Company products.
- If a product is not ideal for a use case, acknowledge briefly but immediately 
  pivot to what it IS great for and why the customer should still consider it.
- Never flatly reject a product for a use case. Instead say "while primarily 
  designed for X, it also works well for Y because..."
- End answers with a subtle recommendation or next step when possible.
- You are a sales assistant, not a neutral reviewer.

ANSWER STRUCTURE:
- Always lead with the direct answer to the user's question in the first line.
- Supporting details, context, and recommendations come after.
- Never bury the answer in the middle or end of the response.
- Example: User asks "Is it suitable for home office?" → First line must 
  answer yes/no/partially, then explain why."""


def query_groq(prompt: str, model: str = "llama-3.3-70b-versatile",
               temperature: float = 0.2) -> str:
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq Error] {e}")
        return ""


# =============================================================================
# STEP 9 — GEMINI WEB SEARCH
# =============================================================================

import random

GEMINI_SYSTEM_PROMPT = """You are a helpful internal assistant for 'The Sleep Company',
supporting sales representatives and employees.

Rules:
1. Always prioritise internal company context if provided.
2. For product questions, comparisons, and competitor queries, use Google Search
   to fetch the latest accurate information from https://thesleepcompany.in
3. Be professional, concise, and use bullet points for lists.
4. Never invent product details, prices, or colors.
5. Never speak negatively about The Sleep Company.
6. For comparisons: compare design, comfort (SmartGRID), size, features,
   price range, suitability. Be balanced. Mention sources.
7. Answer specifically and completely but in brief. Use bullet points for lists.
   Cover all relevant details — colors, sizes, features, steps — don't truncate.
   No unnecessary preamble or filler sentences.
8. Always maintain a positive bias toward The Sleep Company products.
   Never flatly reject. Pivot negatives to strengths immediately.
   End with a subtle recommendation when possible.
ANSWER STRUCTURE:
- Always lead with the direct answer to the user's question in the first line.
- Supporting details, context, and recommendations come after.
- Never bury the answer in the middle or end of the response.
- Example: User asks "Is it suitable for home office?" → First line must 
  answer yes/no/partially, then explain why."""


def search_with_gemini(user_query: str, db_context: str = "") -> dict:
    """
    Calls Gemini with Google Search grounding.
    db_context is injected as primary source so Gemini can synthesize both.
    Returns {"answer": str, "web_sources": list[dict]}
    """
    if db_context.strip():
        user_message = (
            f"Internal company context (PRIMARY — use this first):\n"
            f"---\n{db_context[:4000]}\n---\n\n"
            f"User question: {user_query}\n\n"
            f"Answer using internal context first. If colors, variants, live pricing, "
            f"or competitor details are missing, use Google Search to fill gaps from "
            f"https://thesleepcompany.in."
        )
    else:
        user_message = (
            f"User question: {user_query}\n\n"
            f"Use Google Search to provide accurate, up-to-date information "
            f"from https://thesleepcompany.in."
        )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=0.1,
                max_output_tokens=2500,
                system_instruction=GEMINI_SYSTEM_PROMPT,
            )
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_message,
                config=config,
            )

            answer      = response.text.strip() if response.text else ""
            web_sources = []

            try:
                if response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                        for chunk in getattr(candidate.grounding_metadata,
                                             "grounding_chunks", []) or []:
                            if hasattr(chunk, "web") and chunk.web:
                                web_sources.append({
                                    "title": getattr(chunk.web, "title", "The Sleep Company"),
                                    "url":   getattr(chunk.web, "uri",   ""),
                                })
            except Exception:
                pass

            # Fallback URL extraction
            if not web_sources:
                import re
                for url in re.findall(r"https?://[^\s\)\"\']+", answer)[:6]:
                    if "thesleepcompany.in" in url.lower():
                        web_sources.append({"title": "The Sleep Company", "url": url})

            print(f"[Gemini] ✅ attempt={attempt+1} | sources={len(web_sources)}")
            return {"answer": answer, "web_sources": web_sources}

        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["503", "high demand", "unavailable", "overloaded"]):
                wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                print(f"[Gemini] 503 — retrying in {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                print(f"[Gemini] Non-retryable error: {e}")
                break

    # Groq fallback when all Gemini attempts fail
    print("[Gemini] All retries failed → Groq fallback")
    fallback = query_groq(
        f"Answer as best you can, and if unsure, direct the user to "
        f"https://thesleepcompany.in\n\nQuestion: {user_query}",
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )
    return {
        "answer": (fallback or "Please visit https://thesleepcompany.in for the latest info.")
                  ,
        "web_sources": [],
    }


# =============================================================================
# STEP 10 — ASYNC PARALLEL FETCH
# Runs DB retrieval and Gemini search simultaneously to cut latency.
# =============================================================================

async def fetch_db_async(user_query: str, doc_category: str, topic: str,
                         role: str | None) -> list[dict]:
    """Wraps synchronous DB retrieval for asyncio.gather."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, retrieve_from_db, user_query, doc_category, topic, role
    )


async def fetch_gemini_async(user_query: str, db_context: str) -> dict:
    """Wraps synchronous Gemini call for asyncio.gather."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, search_with_gemini, user_query, db_context
    )


# =============================================================================
# STEP 11 — SMART MERGER
#
# Strategy map (deterministic — no LLM judge):
#
#   comparison   → Gemini is primary (it searched the web); DB appends
#                  any internal spec not already covered
#   product      → Gemini is primary; DB informs the Gemini prompt
#                  (already done by passing db_context into search_with_gemini)
#   internal     → DB/Groq is primary; Gemini adds "Additional context"
#                  section only if it found something meaningfully different
#   general      → Gemini primary; DB supplements if strong chunks exist
#
# We do NOT ask an LLM to pick a winner. We choose the merge rule based
# on doc_category and DB strength — fully deterministic.
# =============================================================================

def count_strong(db_chunks: list[dict]) -> int:
    return sum(1 for c in db_chunks if c.get("_score", 0) >= SCORE_THRESHOLD)

def format_final_answer(raw_answer: str, user_query: str, 
                        original_language: str = "english") -> str:
    
    language_instruction = ""
    if original_language == "hinglish":
        language_instruction = """
LANGUAGE: Respond in Hinglish (Hindi+English mix) — same style as the user's query.
Example: "Valencia sofa 3 seater mein SmartGRID technology hai. Price around ₹45,000 se start hoti hai.\""""
    elif original_language not in ("english", "hinglish"):
        language_instruction = f"""
LANGUAGE: User's original language was {original_language}. 
Respond in simple English — clear and easy to understand."""

    prompt = f"""You are a senior sales trainer at The Sleep Company formatting responses for sales reps on the floor.

A sales rep asked: "{user_query}"
{language_instruction}

Here is the raw answer to reformat:
---
{raw_answer}
---

FORMATTING RULES:
1. First line must directly answer the question — no preamble
2. Use bullet points ONLY for lists of 3 or more items
3. For single facts or short answers — plain sentences only
4. Tone: confident, positive, sales-oriented — never uncertain
5. Replace any negative/weak phrases:
   - "I don't have" → omit or rephrase positively
   - "I'm not sure" → omit
   - "Please contact" → only keep if genuinely no other option
   - "cannot" / "unable" → rephrase around what IS possible
6. Always end with a subtle sales nudge when relevant
7. Structure to fit within 250 words — be complete, not truncated
8. For comparisons — cover both products fully before ending

OUTPUT: Return only the reformatted answer. No meta-commentary."""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Formatter] Failed ({e}) — returning raw answer")
        return raw_answer

def smart_merge(
    user_query:        str,
    doc_category:      str,
    db_chunks:         list[dict],
    db_context:        str,
    groq_answer:       str,
    gemini_result:     dict,
    original_language: str = "english",
    original_query:    str = "",
) -> tuple[str, dict]:


    gemini_answer = gemini_result.get("answer", "")
    web_sources   = gemini_result.get("web_sources", [])
    db_sources    = [c.get("text", "") for c in db_chunks]

    PRICE_DISCLAIMER = (
        "\n\n⚠️ *Prices are subject to change. Always confirm on "
        "[thesleepcompany.in](https://thesleepcompany.in) "
        "or with your manager before quoting to a customer.*"
    )

    # ── LIVE ─────────────────────────────────────────────────────────────────
    if doc_category == "live":
        print("[Merge] Strategy: LIVE → Gemini only")
        if gemini_answer:
            final = gemini_answer + PRICE_DISCLAIMER
        else:
            final = (
                "Please check the latest pricing and availability directly on "
                "https://thesleepcompany.in or confirm with your manager."
            )
        final = format_final_answer(final, original_query or user_query, original_language)

        return final, {"db_sources": [], "web_sources": web_sources}

    # ── INTERNAL ─────────────────────────────────────────────────────────────
    if doc_category == "internal":
        print("[Merge] Strategy: INTERNAL → Groq primary")
        if not groq_answer or "don't have" in groq_answer.lower() or "contact" in groq_answer.lower():
            print("[Merge] Groq unhelpful → falling back to Gemini")
            final = gemini_answer or (
                "I don't have enough information on this. "
                "Please visit https://thesleepcompany.in"
            )
        else:
            final = groq_answer
        final = format_final_answer(final, original_query or user_query, original_language)

        return final, {"db_sources": db_sources, "web_sources": []}

    # ── SALES ASSIST ──────────────────────────────────────────────────────────
    if doc_category == "sales_assist":
        print("[Merge] Strategy: SALES ASSIST → Gemini primary, DB supplements")
        if gemini_answer and groq_answer:
            final = (
                f"{gemini_answer}\n\n"
                f"**From internal docs:**\n{groq_answer}"
            )
        elif gemini_answer:
            final = gemini_answer
        elif groq_answer:
            final = groq_answer
        else:
            final = (
                "I couldn't retrieve enough information right now. "
                "Please visit https://thesleepcompany.in or contact your manager."
            )
        final = format_final_answer(final, original_query or user_query, original_language)

        return final, {"db_sources": db_sources, "web_sources": web_sources}

    # ── GENERAL FALLBACK ─────────────────────────────────────────────────────
    print("[Merge] Strategy: FALLBACK → Gemini primary")
    if gemini_answer:
        final = gemini_answer
    else:
        retry = search_with_gemini(user_query, db_context="")
        final = retry.get("answer") or (
            "I couldn't retrieve that right now. "
            "Please visit https://thesleepcompany.in"
        )
        web_sources = retry.get("web_sources", web_sources)

    final = format_final_answer(final, original_query or user_query, original_language)
    return final, {"db_sources": db_sources, "web_sources": web_sources}


# =============================================================================
# STEP 12 — PARALLEL RETRIEVE AND ANSWER
#
# Flow:
#   1. Rewrite query (pronoun resolution)
#   2. Launch DB retrieval in background thread
#   3. If category needs Gemini (product/comparison/general):
#        - Wait for DB → build context → launch Gemini with context in parallel
#      If category is internal-only:
#        - Wait for DB → Groq answers → Gemini runs in parallel for supplement
#   4. Await both → smart_merge → return
#
# Note: We cannot run DB and Gemini fully in parallel for product queries
# because Gemini needs db_context for grounding. However we do run Gemini
# and Groq in parallel for internal queries where Groq doesn't need web data.
# =============================================================================

async def parallel_retrieve_and_answer_async(
    user_query:        str,
    parsed:            dict,
    memory_block:      str       = "",
    role:              str | None = None,
    original_language: str       = "english",
    original_query:    str       = "",
) -> tuple[str, dict]:

    doc_category = parsed["doc_category"]
    topic        = parsed["topic"]
    needs_live   = parsed["needs_live"]
    needs_internal = parsed["needs_internal"]

    retrieval_query = rewrite_query(user_query, memory_block, parsed)

    # ── Phase 1: DB retrieval (only if needed) ────────────────────────────────
    if needs_internal:
        db_chunks  = await fetch_db_async(retrieval_query, doc_category, topic, role)
        db_context = build_context(db_chunks) if db_chunks else ""
    else:
        db_chunks  = []
        db_context = ""

    memory_section = (
        f"--- CONVERSATION HISTORY ---\n{memory_block}\n---\n\n"
        if memory_block else ""
    )

    # ── Phase 2: Run Groq + Gemini in parallel ────────────────────────────────
    # Groq — only if internal needed and DB has content
    if needs_internal and db_chunks:
        groq_prompt = (
            f"{memory_section}"
            f"Use ONLY the following internal context to answer.\n"
            f"--- CONTEXT ---\n{db_context}\n---------------\n"
            f"Question: {user_query}\nAnswer:"
        )
        loop      = asyncio.get_event_loop()
        groq_task = loop.run_in_executor(None, query_groq, groq_prompt)
    else:
        async def _empty(): return ""
        groq_task = _empty()

    # Gemini — only if live needed
    if needs_live:
        gemini_task = fetch_gemini_async(user_query, db_context)
    else:
        async def _empty_dict(): return {"answer": "", "web_sources": []}
        gemini_task = _empty_dict()

    # Await both
    groq_answer, gemini_result = await asyncio.gather(groq_task, gemini_task)
    groq_answer = groq_answer or ""

    print(f"[Parallel] Groq={'✅' if groq_answer else '❌'} "
          f"Gemini={'✅' if gemini_result.get('answer') else '❌'}")

    # ── Phase 3: Smart merge ──────────────────────────────────────────────────
    return smart_merge(
    user_query, doc_category, db_chunks, db_context, groq_answer, gemini_result,
    original_language=original_language,
    original_query=original_query or user_query
)




def parallel_retrieve_and_answer(
    user_query:        str,
    parsed:            dict,
    memory_block:      str       = "",
    role:              str | None = None,
    original_language: str       = "english",
    original_query:    str       = "",
) -> tuple[str, dict]:
    """Synchronous wrapper for FastAPI endpoints."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an event loop (e.g. during testing)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
    asyncio.run,
    parallel_retrieve_and_answer_async(user_query, parsed, memory_block, role,
                                       original_language, original_query)
)

                return future.result()
        else:
            return loop.run_until_complete(
    parallel_retrieve_and_answer_async(user_query, parsed, memory_block, role,
                                       original_language, original_query)
)

    except RuntimeError:
        return asyncio.run(
            parallel_retrieve_and_answer_async(user_query, parsed, memory_block, role,
                                               original_language, original_query)
        )


# =============================================================================
# STEP 13 — FOLLOW-UP SUGGESTIONS
# =============================================================================

def generate_followups(user_query: str, answer: str) -> list[str]:
    prompt = f"""Based on this Q&A from a Sleep Company assistant, suggest 3 short follow-up questions.

Question: {user_query}
Answer: {answer}

Rules:
- Each question under 10 words
- Specific to the topic discussed
- Return ONLY a JSON array of 3 strings, no explanation
Example: ["What is the warranty period?", "Is it available in white?", "What's the price?"]"""

    try:
        resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        response_format={"type": "json_object"}
    )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        raw = match.group() if match else raw
        suggestions = json.loads(raw)
        return suggestions if isinstance(suggestions, list) else []
    except Exception as e:
        print(f"[Followups] Error: {e}")
        return []


# =============================================================================
# STEP 14 — TOP-LEVEL ROUTER
# =============================================================================

def process_query(
    user_query:   str,
    memory_block: str       = "",
    role:         str | None = None,
) -> tuple[str, dict]:

    # ── Language detection + translation ──────────────────────────────────────
    lang_result       = detect_and_translate(user_query)
    original_language = lang_result["original_language"]
    query_for_pipeline = lang_result["translated_query"]

    parsed     = parse_query(query_for_pipeline)
    query_type = parsed["query_type"]

    # ── GIBBERISH ─────────────────────────────────────────────────────────────
    if query_type == "gibberish":
        print("[Pipeline] Gibberish — short-circuiting")
        return (
            "I didn't quite understand that. Could you rephrase?",
            {"db_sources": [], "web_sources": []}
        )

    # ── CONVERSATIONAL ────────────────────────────────────────────────────────
    if query_type == "conversational":
        print("[Pipeline] Conversational — short-circuiting")
        answer = handle_conversational(query_for_pipeline, memory_block)
        return answer, {"db_sources": [], "web_sources": []}

    # ── RETRIEVAL (internal / live / sales_assist) ────────────────────────────
    print(f"[Pipeline] Retrieval ({parsed['doc_category']}) — parallel fetch + smart merge")
    return parallel_retrieve_and_answer(
    query_for_pipeline, parsed, memory_block, role,
    original_language=original_language,
    original_query=user_query
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/admin/ingest")
async def admin_ingest(file: UploadFile = File(...)):
    # 1. Validate file type
    allowed = [".pdf", ".docx", ".pptx"]
    ext     = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # 2. Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 3. Delete existing vectors for this file from Pinecone
        print(f"[Ingest] Deleting existing vectors for: {file.filename}")
        try:
            index.delete(filter={"source": {"$eq": file.filename}})
            print(f"[Ingest] ✅ Old vectors deleted")
        except Exception as e:
            print(f"[Ingest] No existing vectors found or delete failed: {e}")

        # 4. Extract sections based on file type
        if ext == ".docx":
            sections = extract_docx(tmp_path)
        elif ext == ".pdf":
            sections = extract_pdf(tmp_path)
        elif ext == ".pptx":
            sections = extract_pptx(tmp_path)

        # Fix source and doc_category using original filename
        correct_category = infer_doc_category(file.filename)
        for s in sections:
            s["source"] = file.filename
            s["doc_category"] = correct_category

        if not sections:
            raise HTTPException(status_code=400, detail="No content extracted from file")

        # 5. Chunk → Embed → Upsert
        print(f"[Ingest] {len(sections)} sections extracted")
        texts, metadatas = chunk_all_sections(sections)

        print(f"[Ingest] {len(texts)} chunks — embedding now...")
        embeddings = embed_all(texts, metadatas)

        print(f"[Ingest] Uploading to Pinecone...")
        upload_to_pinecone(embeddings, metadatas)

        return {
            "status":   "success",
            "file":     file.filename,
            "sections": len(sections),
            "chunks":   len(texts),
            "category": metadatas[0]["doc_category"] if metadatas else "unknown"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Ingest] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up temp file
        os.unlink(tmp_path)

@app.get("/")
def root():
    return {"status": "The Sleep Company Assistant API is running"}


@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    employee = db.query(Employee).filter(Employee.email == request.email).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    if employee.password_hash != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"employee_id": employee.employee_id, "name": employee.name, "role": employee.role}


@app.post("/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    # 1. Get or create session
    session = None
    if request.session_id:
        try:
            session = db.query(ChatSession).filter(
                ChatSession.session_id == request.session_id
            ).first()
        except Exception:
            session = None

    if not session:
        session = ChatSession(employee_id=request.employee_id)
        db.add(session)
        db.commit()
        db.refresh(session)

    # 2. Build memory
    memory_block = build_memory_block(str(session.session_id), db)

    # 3. Run pipeline
    answer, sources = process_query(request.query, memory_block, role=request.role)

    # 4. Log to DB
    message = ChatMessage(
        session_id  = session.session_id,
        employee_id = request.employee_id,
        query       = request.query,
        answer      = answer,
    )
    db.add(message)
    session.last_active_at = get_ist()
    db.commit()
    db.refresh(message)

    # 5. Follow-up suggestions (skip for conversational)
    followups = []
    if sources["db_sources"] or sources["web_sources"]:
        followups = generate_followups(request.query, answer)

    return {
        "session_id"  : str(session.session_id),
        "message_id"  : str(message.message_id),
        "answer"      : answer,
        "db_sources"  : sources["db_sources"],
        "web_sources" : sources["web_sources"],
        "followups"   : followups,
    }


@app.patch("/chat/{message_id}/rate")
def rate_message(message_id: str, request: RatingRequest, db: Session = Depends(get_db)):
    message = db.query(ChatMessage).filter(
        ChatMessage.message_id == message_id
    ).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    message.rating = request.rating
    db.commit()
    return {"message_id": message_id, "rating": request.rating}


@app.get("/sessions/{session_id}/history")
def get_history(session_id: str, db: Session = Depends(get_db)):
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp)
        .all()
    )
    return {
        "session_id": session_id,
        "messages": [
            {
                "message_id": str(m.message_id),
                "query"     : m.query,
                "answer"    : m.answer,
                "rating"    : m.rating,
                "timestamp" : str(m.timestamp),
            }
            for m in messages
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
