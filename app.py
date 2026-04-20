import os
import time
import json
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
                             # increased from 3 — better conversational coherence
DB_STRONG_THRESHOLD = 5      # min strong DB chunks needed to skip web search
                             # increased from 3 — avoids premature web fallback
SCORE_THRESHOLD     = 0.25   # min Pinecone score to count a chunk as "strong"
                             # lowered from 0.60 — bge-base scores cluster 0.45–0.65

# =============================================================================
# CLIENTS
# =============================================================================
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
}
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "sales-chatbot"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")   # ← You can comment this out now

# New: Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

print("HF_API_KEY :", "✅" if HF_API_KEY else "❌ MISSING")
print("PINECONE_KEY :", "✅" if PINECONE_API_KEY else "❌ MISSING")
print("GROQ_API_KEY :", "✅" if GROQ_API_KEY else "❌ MISSING")
print("GEMINI_API_KEY :", "✅" if GEMINI_API_KEY else "❌ MISSING")
# print("ANTHROPIC_API_KEY :", "✅" if ANTHROPIC_API_KEY else "❌ MISSING")  # optional

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
groq_client = Groq(api_key=GROQ_API_KEY)

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
    role:        str | None = None   # "sales" | "employee" — used for retrieval filtering

class RatingRequest(BaseModel):
    rating: str

class LoginRequest(BaseModel):
    email:    str
    password: str


# =============================================================================
# HELPER: Detect Product Queries
# =============================================================================
def is_product_query(parsed: dict, user_query: str) -> bool:
    """Return True if this should go to website (Gemini)"""
    query_type = parsed.get("query_type", "")
    doc_category = parsed.get("doc_category", "general")
    topic = parsed.get("topic", "").lower()
    q = user_query.lower()

    # Strong signals
    product_keywords = [
        "sofa", "mattress", "pillow", "recliner", "bed", "chair", "valencia",
        "color", "colour", "colors", "colours", "shade", "option", "variant",
        "available in", "how many", "price", "warranty", "dimension", "size",
        "recommend", "best", "which", "catalog"
    ]

    if query_type == "retrieval" and doc_category == "product":
        return True

    if any(keyword in q for keyword in product_keywords):
        return True

    if "valencia" in q or "product" in doc_category:
        return True

    return False

# =============================================================================
# STEP 1 — QUERY PARSER
# Classifies intent. Defaults aggressively to "retrieval" to prevent
# hallucination from product/policy queries being routed to LLM-only paths.
# =============================================================================

def parse_query(user_query: str) -> dict:
    prompt = f"""You are a query classifier for a 'The Sleep Company' assistant chatbot.
The assistant has access to internal documents including:
  - Product catalogs and specs (sofas, mattresses, pillows, recliners)
  - Company SOPs (Standard Operating Procedures)
  - HR and company policies
  - Training manuals and guides
  - Pricing documents

Given the user query below, return ONLY a valid JSON object with these fields:

  "query_type"   : one of ["conversational", "informational", "retrieval"]

  STRICT RULES for query_type:
  - "conversational" ONLY IF: pure greeting ("hi", "hello", "thanks", "bye"),
    small talk, OR explicitly asks what you can do ("what can you help with")
  - "informational" ONLY IF: general world knowledge with ZERO connection
    to sleep, sofas, furniture, mattresses, pillows, or company operations
  - "retrieval" FOR EVERYTHING ELSE — when in doubt, ALWAYS choose retrieval
  - Short or vague product queries ("sofa recommendations", "mattress options",
    "best pillow", "what sofas do you have") → ALWAYS retrieval
  - Any query mentioning a product name, category, price, warranty, SOP,
    policy, or training → ALWAYS retrieval

  "doc_category" : one of ["product", "policy", "sop", "training", "pricing", "faq", "general"]
                   Only relevant when query_type is "retrieval". Default "general" otherwise.

  "topic"        : short phrase (max 6 words) describing the specific subject.
                   Only relevant when query_type is "retrieval". Empty string otherwise.

Return ONLY the JSON object, no explanation, no markdown.

User query: {user_query}"""

    try:
        response = groq_client.chat.completions.create(
            model      = "llama-3.1-8b-instant",
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0,
            max_tokens = 80,
        )
        raw    = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        valid_types      = {"conversational", "informational", "retrieval"}
        valid_categories = {"product", "policy", "sop", "training", "pricing", "faq", "general"}

        query_type   = parsed.get("query_type", "retrieval")
        doc_category = parsed.get("doc_category", "general")
        topic        = parsed.get("topic", "")

        if query_type   not in valid_types:      query_type   = "retrieval"
        if doc_category not in valid_categories: doc_category = "general"
        if not isinstance(topic, str):           topic        = ""

        # Safety net: if query contains product signals, force retrieval
        product_signals = [
            "sofa", "mattress", "pillow", "recliner", "bed", "chair",
            "recommend", "suggest", "option", "catalog", "price", "warranty",
            "sop", "policy", "leave", "training", "onboarding",
        ]
        if query_type != "retrieval" and any(
            w in user_query.lower() for w in product_signals
        ):
            print(f"[Parser] Overriding {query_type} → retrieval (product signal detected)")
            query_type = "retrieval"

        print(f"[Parser] type={query_type} | category={doc_category} | topic={topic!r}")
        return {"query_type": query_type, "doc_category": doc_category, "topic": topic}

    except Exception as e:
        print(f"[Parser] Failed ({e}) — defaulting to retrieval")
        return {"query_type": "retrieval", "doc_category": "general", "topic": ""}


# =============================================================================
# STEP 2 — QUERY REWRITER
# Resolves pronouns and follow-up references before hitting Pinecone.
# "What's its warranty?" → "What's the Valencia sofa warranty?"
# This is the single biggest fix for conversational follow-up failures.
# =============================================================================

def rewrite_query(user_query: str, memory_block: str) -> str:
    if not memory_block:
        return user_query

    prompt = f"""Rewrite the user's question as a fully self-contained search query.
Resolve any pronouns, references like "it", "that one", "the same", or follow-up
references using the conversation history below.
Return ONLY the rewritten query as a single sentence. Nothing else.

Conversation history:
{memory_block}

User question: {user_query}
Rewritten query:"""

    try:
        response = groq_client.chat.completions.create(
            model      = "llama-3.1-8b-instant",
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0,
            max_tokens = 60,
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten and rewritten != user_query:
            print(f"[Rewriter] '{user_query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[Rewriter] Failed ({e}) — using original query")
        return user_query


# =============================================================================
# STEP 3 — CONVERSATIONAL HANDLER
# Short-circuits for greetings and capability questions.
# =============================================================================

# =============================================================================
# STEP 3 — CONVERSATIONAL HANDLER (Dynamic & Context-Aware)
# =============================================================================
def handle_conversational(user_query: str, memory_block: str = "") -> str:
    """Handles greetings intelligently by referring back to recent conversation dynamically."""
    
    if not memory_block:
        # Pure greeting - no history
        prompt = f"""You are a friendly and professional assistant for 'The Sleep Company'.
User sent a greeting. Respond warmly in 2-3 sentences.
Mention you can help with product information, recommendations, SOPs, policies, and pricing.

User message: {user_query}"""

    else:
        # There is recent conversation history → make it contextual
        prompt = f"""You are a helpful assistant for 'The Sleep Company'.

Recent conversation:
{memory_block}

User just said: "{user_query}"

Instructions:
- Greet the user warmly and naturally.
- If the previous conversation was about products (sofa, recliner, mattress, recommendation, price, colors, etc.), politely refer back to it.
- Do not repeat the full previous answer. Instead, offer to continue helping on that topic or ask what else they need.
- Keep the response natural, concise (3-5 sentences max), and professional.
- Never make up prices, colors, or specs.

Respond directly to the user:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Conversational] Error: {e}")
        # Safe generic fallback
        return "Good morning! 👋 I'm happy to help you today. How can I assist you with our products, policies, or anything else?"


# =============================================================================
# STEP 4 — EMBEDDING
# Context prefix MUST match ingest.py's build_embed_text() exactly.
# Format: "Category: <doc_category>. Heading: <heading>.\n<query>"
# =============================================================================

def build_query_embed_text(query: str, doc_category: str, topic: str) -> str:
    heading_hint = topic if topic else "user query"
    return (
        f"Category: {doc_category}. "
        f"Heading: {heading_hint}.\n"
        f"{query}"
    )


def get_embedding(text: str) -> list[float]:
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    result   = response.json()

    if isinstance(result, dict):
        if "error" in result:
            raise Exception(f"HF API Error: {result['error']}")
        if "estimated_time" in result:
            raise Exception("model_loading")

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
# Role-based filtering: sales reps only see customer-facing doc types.
# Falls back to unfiltered if filtered results are too sparse.
# =============================================================================

# What each role is allowed to retrieve
ROLE_CATEGORY_ALLOW = {
    "sales":    {"product", "pricing", "faq", "general"},
    "employee": {"product", "policy", "sop", "training", "pricing", "faq", "general"},
}

def retrieve_from_db(
    user_query:   str,
    doc_category: str,
    topic:        str,
    role:         str | None = None,
    top_k:        int = SOURCES_ACCESSED,
) -> list[dict]:
    embed_text = build_query_embed_text(user_query, doc_category, topic)
    embedding  = get_embedding_with_retry(embed_text)

    # Build filter: combine doc_category + role restrictions
    pinecone_filter = {}

    allowed_categories = ROLE_CATEGORY_ALLOW.get(role, None) if role else None

    if doc_category and doc_category != "general":
        if allowed_categories and doc_category not in allowed_categories:
            # Queried category not allowed for this role — use role filter only
            print(f"[Retrieval] Role '{role}' blocked category '{doc_category}' — using role filter")
            pinecone_filter = {"doc_category": {"$in": list(allowed_categories)}}
        else:
            pinecone_filter = {"doc_category": {"$eq": doc_category}}
    elif allowed_categories and role == "sales":
        # Sales rep with general query — restrict to allowed categories
        pinecone_filter = {"doc_category": {"$in": list(allowed_categories)}}

    results = index.query(
        vector          = embedding,
        top_k           = top_k,
        include_metadata= True,
        filter          = pinecone_filter if pinecone_filter else None,
    )
    matches = results.get("matches", [])

    # Fallback: if filter is too narrow, retry without it
    if len(matches) < 2 and pinecone_filter:
        print(f"[Retrieval] Filter too narrow — falling back to unfiltered")
        results = index.query(
            vector          = embedding,
            top_k           = top_k,
            include_metadata= True,
        )
        matches = results.get("matches", [])

    valid = []
    for m in matches:
        if m["metadata"].get("text"):
            chunk          = dict(m["metadata"])
            chunk["_score"] = m.get("score", 0.0)
            valid.append(chunk)

    print(f"[Retrieval] {len(valid)} chunks (filter={pinecone_filter or 'none'})")
    return valid


# =============================================================================
# STEP 6 — GEMINI WEB SEARCH (replaces Claude)
# Uses Gemini's native Grounding with Google Search tool.
# Prioritises internal DB context and uses web search only when needed.
# =============================================================================
# =============================================================================
# STEP 6 — GEMINI WEB SEARCH (Enhanced for Product Queries)
# Strongly encourages fetching fresh product info from the official website.
# =============================================================================
import time
import random

# =============================================================================
# STEP 6 — GEMINI WEB SEARCH (Robust Version with Retry + Strong Product Focus)
# =============================================================================
def search_with_gemini(
    user_query: str,
    db_context: str = "",
) -> dict:
    """
    Uses Gemini with Google Search grounding.
    Includes retry on 503 errors + graceful fallback to Groq.
    Optimized for product queries (colors, specs, etc.).
    """
    system_prompt = (
        "You are a helpful internal assistant for 'The Sleep Company', supporting sales "
        "representatives and employees.\n\n"
        "Rules:\n"
        "1. Always prioritize the internal company context if provided.\n"
        "2. For product questions (colors, variants, price, warranty, dimensions, recommendations, "
        "'best sofa', etc.), actively use Google Search to fetch the **latest accurate information** "
        "from the official website: https://thesleepcompany.in\n"
        "3. Clearly separate sources: Use 'From internal documents:' and 'From our website:' when needed.\n"
        "4. Be professional, concise, and use bullet points for lists (especially colors).\n"
        "5. Never invent or hallucinate product details, prices, or colors.\n"
        "6. Never speak negatively about The Sleep Company.\n"
    )

    # Build user message
    if db_context.strip():
        user_message = (
            f"Internal company context (PRIMARY source):\n"
            f"---\n{db_context[:4000]}\n---\n\n"
            f"User question: {user_query}\n\n"
            f"Answer using internal context first. "
            f"If details like colors, variants, or specs are missing/incomplete, "
            f"use Google Search to get the latest information directly from thesleepcompany.in."
        )
    else:
        user_message = (
            f"User question: {user_query}\n\n"
            f"This is likely a product query. Use Google Search to provide accurate, "
            f"up-to-date information from https://thesleepcompany.in."
        )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            grounding_tool = types.Tool(google_search=types.GoogleSearch())

            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=0.1,          # Low for factual product answers
                max_output_tokens=1200,
                system_instruction=system_prompt,
            )

            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_message,
                config=config,
            )

            answer = response.text.strip() if response.text else ""

            # Extract grounding sources
            web_sources = []
            try:
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        for chunk in getattr(candidate.grounding_metadata, 'grounding_chunks', []) or []:
                            if hasattr(chunk, 'web') and chunk.web:
                                web_sources.append({
                                    "title": getattr(chunk.web, 'title', 'The Sleep Company'),
                                    "url": getattr(chunk.web, 'uri', '')
                                })
            except Exception:
                pass

            # Fallback: extract official website URLs from answer
            if not web_sources:
                import re
                url_pattern = re.compile(r'https?://[^\s\)\"\']+')
                urls_found = url_pattern.findall(answer)
                for url in urls_found[:6]:
                    if "thesleepcompany.in" in url.lower():
                        web_sources.append({"title": "The Sleep Company Website", "url": url})

            print(f"[Gemini Search] Success on attempt {attempt+1} | Sources={len(web_sources)}")
            return {"answer": answer, "web_sources": web_sources}

        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["503", "high demand", "unavailable", "overloaded"]):
                wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                print(f"[Gemini Search] 503 High Demand - retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            else:
                print(f"[Gemini Search] Non-retryable error: {e}")
                break

    # Final fallback when all Gemini attempts fail
    print("[Gemini Search] All retries failed → falling back to Groq LLM")
    try:
        fallback_prompt = f"""You are a helpful assistant for The Sleep Company.
Answer this product-related question as accurately as possible.
If you don't have exact details, politely direct the user to check the official website.

Question: {user_query}"""

        fallback_answer = query_llm(fallback_prompt, model="llama-3.3-70b-versatile", temperature=0.2)
        
        return {
            "answer": fallback_answer + "\n\n(Note: Our search service is temporarily busy. Please verify the latest details on https://thesleepcompany.in)",
            "web_sources": []
        }
    except Exception:
        return {
            "answer": "I'm currently experiencing high load on our search service. Please try again in a moment or visit https://thesleepcompany.in for the latest product information.",
            "web_sources": []
        }

# =============================================================================
# STEP 7 — CONTEXT BUILDER
# Assembles DB chunks into a single text block for the LLM prompt.
# Token budget: caps total context at ~3000 chars to prevent dilution.
# =============================================================================

MAX_CONTEXT_CHARS = 3000

def build_context(db_chunks: list[dict]) -> str:
    parts        = []
    total_chars  = 0

    for chunk in db_chunks:
        source   = chunk.get("source", "internal document")
        heading  = chunk.get("heading", "")
        category = chunk.get("doc_category", "")
        text     = chunk.get("text", "")
        score    = chunk.get("_score", 0.0)

        label = f"[{category.upper()} — {source}] {heading}"
        entry = f"{label}\n{text}"

        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            print(f"[Context] Token budget reached — stopped at {len(parts)} chunks")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n\n---\n\n".join(parts)


# =============================================================================
# STEP 8 — CONVERSATION MEMORY
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
# STEP 9 — GROQ LLM (used for DB-only answers)
# =============================================================================

def query_llm(prompt: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2) -> str:
    try:
        response = groq_client.chat.completions.create(
            model    = model,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an internal company assistant for sales representatives "
                        "and employees of 'The Sleep Company'.\n\n"
                        "Your role:\n"
                        "- Help with product recommendations\n"
                        "- Answer SOP, policy, and training-related questions\n"
                        "- Assist sales reps in customer conversations\n\n"
                        "CRITICAL RULES:\n"
                        "- Use ONLY the provided context\n"
                        "- Do NOT use outside knowledge\n"
                        "- Do NOT hallucinate product names, prices, or specs\n\n"
                        "If information is missing:\n"
                        "- Say: 'I don't have that information. Please contact the relevant team.'\n\n"
                        "STYLE:\n"
                        "- Professional, concise, point-wise\n"
                        "- Plain text only\n\n"
                        "COMPANY RULE:\n"
                        "- Never portray the company negatively\n"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature = temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM Error] {e}")
        return "I'm facing a temporary issue. Please try again."


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
        response = groq_client.chat.completions.create(
            model      = "llama-3.1-8b-instant",
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0.4,
        )
        raw         = response.choices[0].message.content.strip()
        suggestions = json.loads(raw)
        return suggestions if isinstance(suggestions, list) else []
    except Exception as e:
        print(f"[Followups] Error: {e}")
        return []


# =============================================================================
# STEP 10 — RETRIEVE AND ANSWER
#
# Three-path routing based on DB result quality:
#
#   Path A — DB strong (>= DB_STRONG_THRESHOLD strong chunks):
#     → Build context from DB chunks
#     → Answer with Groq LLM (fast, cheap, grounded in internal docs)
#     → No web search
#
#   Path B — DB weak + web needed:
#     → Pass weak DB context to Claude
#     → Claude searches the web AND merges with internal context
#     → Returns Claude's synthesised answer
#
#   Path C — DB empty:
#     → Pure Claude web search, no internal context
#     → Fallback message if Claude also fails
# =============================================================================

# =============================================================================
# STEP 10 — RETRIEVE AND ANSWER (Updated for Product Queries)
# =============================================================================
def retrieve_and_answer(
    user_query: str,
    parsed: dict,
    memory_block: str = "",
    role: str | None = None,
) -> tuple[str, dict]:
    """
    Returns (answer, sources_dict)
    """
    doc_category = parsed["doc_category"]
    topic = parsed["topic"]

    # Rewrite query for better retrieval
    retrieval_query = rewrite_query(user_query, memory_block)

    # --- Always query DB first (for context) ---
    db_chunks = retrieve_from_db(retrieval_query, doc_category, topic, role=role)
    strong_chunks = [c for c in db_chunks if c.get("_score", 0.0) >= SCORE_THRESHOLD]

    print(f"[Pipeline] {len(strong_chunks)} strong chunks (threshold={SCORE_THRESHOLD})")

    # ===================================================================
    # NEW LOGIC: PRODUCT QUERIES → ALWAYS GO TO WEBSITE (Gemini)
    # ===================================================================
    if is_product_query(parsed, user_query):
        print(f"[Pipeline] Product query detected → forcing Gemini with web search")
        db_context = build_context(db_chunks) if db_chunks else ""
        
        result = search_with_gemini(user_query, db_context=db_context)
        
        if result["answer"]:
            sources = {
                "db_sources": [c.get("text", "") for c in db_chunks],
                "web_sources": result["web_sources"],
            }
            return result["answer"], sources
        else:
            # Fallback
            return (
                "I don't have enough information right now. Please check our website or contact the team.",
                {"db_sources": [], "web_sources": []}
            )

    # ===================================================================
    # NON-PRODUCT QUERIES → Keep original logic (DB strong → Groq)
    # ===================================================================
    if len(strong_chunks) >= DB_STRONG_THRESHOLD:
        print("[Pipeline] Path A — DB strong, answering with Groq")
        context = build_context(db_chunks)
        memory_section = (
            f"--- CONVERSATION HISTORY ---\n{memory_block}\n----------------------------\n"
            if memory_block else ""
        )
        prompt = f"""{memory_section}Use ONLY the following context to answer the question.
The context may include product information, company policies, SOPs, or training material.
If the answer is not present, say so clearly.
--- CONTEXT ---
{context}
---------------
Question: {user_query}
Answer:"""
        answer = query_llm(prompt)
        sources = {
            "db_sources": [c.get("text", "") for c in db_chunks],
            "web_sources": [],
        }
        return answer, sources

    # Path B — DB weak but some context available
    elif db_chunks:
        print("[Pipeline] Path B — DB weak, handing to Gemini with web search")
        db_context = build_context(db_chunks)
        result = search_with_gemini(user_query, db_context=db_context)
        if result["answer"]:
            sources = {
                "db_sources": [c.get("text", "") for c in db_chunks],
                "web_sources": result["web_sources"],
            }
            return result["answer"], sources

    # Path C — No DB chunks
    else:
        print("[Pipeline] Path C — DB empty, pure Gemini web search")
        result = search_with_gemini(user_query)
        if result["answer"]:
            return result["answer"], {"db_sources": [], "web_sources": result["web_sources"]}

    # Final fallback
    return (
        "I don't have enough information to answer that. Please contact the relevant team.",
        {"db_sources": [], "web_sources": []},
    )

# =============================================================================
# STEP 11 — PROCESS QUERY (top-level router)
# =============================================================================

def process_query(
    user_query:   str,
    memory_block: str = "",
    role:         str | None = None,
) -> tuple[str, dict]:

    parsed     = parse_query(user_query)
    query_type = parsed["query_type"]

    # --- Conversational: greetings, capability questions ---
    if query_type == "conversational":
        print("[Pipeline] Conversational — short-circuiting")
        answer = handle_conversational(user_query)
        return answer, {"db_sources": [], "web_sources": []}

    # --- Informational: pure general knowledge, no company connection ---
    if query_type == "informational":
        print("[Pipeline] Informational — LLM only (no retrieval)")
        prompt = f"Answer this general question concisely and professionally:\n\n{user_query}"
        answer = query_llm(prompt, model="llama-3.1-8b-instant", temperature=0.3)
        return answer, {"db_sources": [], "web_sources": []}

    # --- Retrieval: full pipeline ---
    return retrieve_and_answer(user_query, parsed, memory_block, role=role)


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

    return {
        "employee_id": employee.employee_id,
        "name":        employee.name,
        "role":        employee.role,
    }


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

    # 3. Run pipeline — pass role for retrieval filtering
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

    # 5. Generate follow-ups (skip for conversational — no sources)
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