import os
import time
import json
import asyncio
import requests
from dotenv import load_dotenv

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

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

SOURCES_ACCESSED    = 10     # top_k chunks pulled from Pinecone per query
MEMORY_TURNS        = 5      # how many past Q&A pairs to include in the LLM prompt
DB_STRONG_THRESHOLD = 4      # min strong DB chunks to consider DB context "rich"
SCORE_THRESHOLD     = 0.25   # min Pinecone score to count a chunk as "strong"
MAX_CONTEXT_CHARS   = 3000   # cap on DB context fed to LLM

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

# =============================================================================
# STEP 1 — QUERY PARSER
# =============================================================================

def parse_query(user_query: str) -> dict:
    """
    Classifies the query into:
      query_type   : conversational | informational | retrieval
      doc_category : product | policy | sop | training | pricing | faq |
                     comparison | general
      topic        : short phrase for embedding context

    'comparison' is a new category — triggers Gemini-first merge.
    """
    prompt = f"""You are a query classifier for 'The Sleep Company' assistant chatbot.
The assistant has access to:
  - Product catalogs and specs (sofas, mattresses, pillows, recliners)
  - Company SOPs, HR and company policies
  - Training manuals, pricing documents

Given the user query, return ONLY a valid JSON object with:

  "query_type": one of ["conversational", "informational", "retrieval"]
    - "conversational" ONLY for: pure greetings, small talk, capability questions
    - "informational"  ONLY for: general world knowledge with ZERO connection
      to sleep, furniture, or company operations
    - "retrieval" FOR EVERYTHING ELSE — when in doubt always choose retrieval

  "doc_category": one of ["product", "policy", "sop", "training", "pricing",
                           "faq", "comparison", "general"]
    - "comparison" when the query compares two products/brands OR asks
      which is better (e.g. "Valencia vs Galway", "better than Wakefit")
    - "product"    for specs, colors, features, recommendations of a single item
    - Use others as appropriate

  "topic": short phrase (max 6 words). Empty string for non-retrieval.

Return ONLY the JSON object.

User query: {user_query}"""

    try:
        resp   = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        raw    = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)

        valid_types = {"conversational", "informational", "retrieval"}
        valid_cats  = {"product", "policy", "sop", "training", "pricing",
                       "faq", "comparison", "general"}

        query_type   = parsed.get("query_type", "retrieval")
        doc_category = parsed.get("doc_category", "general")
        topic        = parsed.get("topic", "")

        if query_type   not in valid_types: query_type   = "retrieval"
        if doc_category not in valid_cats:  doc_category = "general"
        if not isinstance(topic, str):      topic        = ""

        q_lower = user_query.lower()

        # Comparison override — highest priority
        if any(sig in q_lower for sig in COMPARISON_SIGNALS):
            query_type   = "retrieval"
            doc_category = "comparison"
            print(f"[Parser] Comparison signal detected → doc_category=comparison")

        # Product/internal signal override
        elif query_type != "retrieval" and any(sig in q_lower for sig in PRODUCT_SIGNALS):
            print(f"[Parser] Product signal → overriding {query_type} to retrieval")
            query_type = "retrieval"

        print(f"[Parser] type={query_type} | category={doc_category} | topic={topic!r}")
        return {"query_type": query_type, "doc_category": doc_category, "topic": topic}

    except Exception as e:
        print(f"[Parser] Failed ({e}) — defaulting to retrieval")
        return {"query_type": "retrieval", "doc_category": "general", "topic": ""}


# =============================================================================
# STEP 2 — QUERY REWRITER
# =============================================================================

def rewrite_query(user_query: str, memory_block: str) -> str:
    """Resolves pronouns and follow-up references before hitting Pinecone."""
    if not memory_block:
        return user_query

    prompt = f"""Rewrite the user's question as a fully self-contained search query.
Resolve any pronouns or references using the conversation history.
Return ONLY the rewritten query as a single sentence.

Conversation history:
{memory_block}

User question: {user_query}
Rewritten query:"""

    try:
        resp      = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        rewritten = resp.choices[0].message.content.strip()
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
Respond warmly in 2-3 sentences. Mention you can help with products,
recommendations, SOPs, policies, and pricing.
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

def retrieve_from_db(
    user_query:   str,
    doc_category: str,
    topic:        str,
    role:         str | None = None,
    top_k:        int = SOURCES_ACCESSED,
) -> list[dict]:
    # Use "product" embedding context for comparisons (best semantic match)
    embed_category = "product" if doc_category == "comparison" else doc_category
    embed_text     = build_query_embed_text(user_query, embed_category, topic)
    embedding      = get_embedding_with_retry(embed_text)

    allowed_categories = ROLE_CATEGORY_ALLOW.get(role) if role else None
    pinecone_filter    = {}

    if doc_category == "comparison":
        # Pull product chunks from both items being compared
        pinecone_filter = {"doc_category": {"$eq": "product"}}
    elif doc_category and doc_category != "general":
        if allowed_categories and doc_category not in allowed_categories:
            print(f"[Retrieval] Role '{role}' blocked '{doc_category}' — using role filter")
            pinecone_filter = {"doc_category": {"$in": list(allowed_categories)}}
        else:
            pinecone_filter = {"doc_category": {"$eq": doc_category}}
    elif allowed_categories and role == "sales":
        pinecone_filter = {"doc_category": {"$in": list(allowed_categories)}}

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
- If information is missing say: 'I don't have that detail in our internal docs.'

STYLE: Professional, concise, point-wise. Plain text only.
COMPANY RULE: Never portray the company negatively."""


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
   price range, suitability. Be balanced. Mention sources."""


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
                max_output_tokens=1200,
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
                  + "\n\n_(Search service temporarily busy — please verify details online.)_",
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


def smart_merge(
    user_query:    str,
    doc_category:  str,
    db_chunks:     list[dict],
    db_context:    str,
    groq_answer:   str,
    gemini_result: dict,
) -> tuple[str, dict]:
    """
    Deterministically merges DB and Gemini answers based on query category.
    Returns (final_answer, sources_dict).
    """
    gemini_answer = gemini_result.get("answer", "")
    web_sources   = gemini_result.get("web_sources", [])
    db_sources    = [c.get("text", "") for c in db_chunks]
    strong_count  = count_strong(db_chunks)

    # ── COMPARISON queries ────────────────────────────────────────────────────
    # Gemini searched the web and has the most complete picture.
    # Groq/DB is not used as primary; DB context was already given to Gemini.
    if doc_category == "comparison":
        print("[Merge] Strategy: COMPARISON → Gemini primary")
        if not gemini_answer:
            # Gemini failed entirely
            final = (
                "I couldn't retrieve a live comparison right now. "
                "Please visit https://thesleepcompany.in or contact the sales team."
            )
        else:
            final = gemini_answer
        return final, {"db_sources": db_sources, "web_sources": web_sources}

    # ── INTERNAL queries (policy, sop, training) ──────────────────────────────
    # DB/Groq is authoritative. Gemini supplements only if it adds new info.
    if doc_category in INTERNAL_CATEGORIES:
        print(f"[Merge] Strategy: INTERNAL ({doc_category}) → Groq primary")
        if not groq_answer:
            # DB had nothing; fall back to Gemini
            final = gemini_answer or (
                "I don't have that information in our internal documents. "
                "Please contact the relevant team."
            )
        elif gemini_answer and gemini_answer.strip() and len(gemini_answer) > 80:
            # Gemini found supplementary public info — append it
            final = (
                f"{groq_answer}\n\n"
                f"**Additional context (public sources):**\n{gemini_answer}"
            )
        else:
            final = groq_answer
        return final, {"db_sources": db_sources, "web_sources": web_sources}

    # ── PRODUCT / PRICING / FAQ / GENERAL ────────────────────────────────────
    # Gemini is primary (live product data, colors, prices).
    # If Gemini failed, fall back to Groq only if DB was strong.
    print(f"[Merge] Strategy: PRODUCT/GENERAL → Gemini primary")

    # ── PRODUCT / PRICING / FAQ / GENERAL ────────────────────────────────────
    if gemini_answer:
        final = gemini_answer
    else:
        # Gemini returned empty — retry with pure web search, no DB context
        print("[Merge] Gemini empty → retrying with pure web search")
        retry = search_with_gemini(user_query, db_context="")
        final = retry.get("answer") or (
            "I couldn't retrieve that right now. "
            "Please visit https://thesleepcompany.in"
        )
        web_sources = retry.get("web_sources", web_sources)

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
    user_query:   str,
    parsed:       dict,
    memory_block: str       = "",
    role:         str | None = None,
) -> tuple[str, dict]:

    doc_category = parsed["doc_category"]
    topic        = parsed["topic"]

    retrieval_query = rewrite_query(user_query, memory_block)

    # ── Phase 1: DB retrieval (always needed) ────────────────────────────────
    db_chunks  = await fetch_db_async(retrieval_query, doc_category, topic, role)
    db_context = build_context(db_chunks) if db_chunks else ""

    memory_section = (
        f"--- CONVERSATION HISTORY ---\n{memory_block}\n---\n\n"
        if memory_block else ""
    )

    # ── Phase 2: Run Groq + Gemini in parallel ───────────────────────────────
    # Groq task — only meaningful when DB has content
    if db_chunks:
        groq_prompt = (
            f"{memory_section}"
            f"Use ONLY the following internal context to answer.\n"
            f"--- CONTEXT ---\n{db_context}\n---------------\n"
            f"Question: {user_query}\nAnswer:"
        )
        loop      = asyncio.get_event_loop()
        groq_task = loop.run_in_executor(None, query_groq, groq_prompt)
    else:
        groq_task = asyncio.coroutine(lambda: "")()

    # Gemini task — always runs (it has db_context for grounding)
    gemini_task = fetch_gemini_async(user_query, db_context)

    # Await both
    groq_answer, gemini_result = await asyncio.gather(groq_task, gemini_task)

    groq_answer = groq_answer or ""

    print(f"[Parallel] Groq={'✅' if groq_answer else '❌'} "
          f"Gemini={'✅' if gemini_result.get('answer') else '❌'}")

    # ── Phase 3: Smart merge ─────────────────────────────────────────────────
    return smart_merge(
        user_query, doc_category, db_chunks, db_context, groq_answer, gemini_result
    )


def parallel_retrieve_and_answer(
    user_query:   str,
    parsed:       dict,
    memory_block: str       = "",
    role:         str | None = None,
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
                    parallel_retrieve_and_answer_async(user_query, parsed, memory_block, role)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                parallel_retrieve_and_answer_async(user_query, parsed, memory_block, role)
            )
    except RuntimeError:
        return asyncio.run(
            parallel_retrieve_and_answer_async(user_query, parsed, memory_block, role)
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
        )
        raw         = resp.choices[0].message.content.strip()
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

    parsed     = parse_query(user_query)
    query_type = parsed["query_type"]

    if query_type == "conversational":
        print("[Pipeline] Conversational — short-circuiting")
        answer = handle_conversational(user_query, memory_block)
        return answer, {"db_sources": [], "web_sources": []}

    if query_type == "informational":
        print("[Pipeline] Informational — Groq only")
        answer = query_groq(
            f"Answer concisely:\n{user_query}",
            model="llama-3.1-8b-instant",
        )
        return answer, {"db_sources": [], "web_sources": []}

    # All retrieval queries (product / comparison / policy / sop / training / …)
    print(f"[Pipeline] Retrieval ({parsed['doc_category']}) — parallel fetch + smart merge")
    return parallel_retrieve_and_answer(user_query, parsed, memory_block, role)


# =============================================================================
# API ENDPOINTS
# =============================================================================

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