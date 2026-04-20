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

from database import get_db
from models import ChatSession, ChatMessage, Employee, get_ist

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

SOURCES_ACCESSED     = 10      # top_k chunks pulled from Pinecone per query
MEMORY_TURNS         = 3      # how many past Q&A pairs to include in the LLM prompt
MAX_RETRIES          = 2      # validation retry attempts before returning best effort
DB_STRONG_THRESHOLD  = 5      # min DB chunks needed to skip web search
SCORE_THRESHOLD      = 0.20   # min Pinecone score to count a chunk as "strong"
WEB_SCORE_THRESHOLD  = 0.60   # min Tavily score to keep a web result
MAX_WEB_RESULTS      = 2      # cap on injected web results

# =============================================================================
# CHANGE 1 — DOMAIN ALLOWLIST FOR WEB SEARCH
# Only these domains will be searched via Tavily.
# Add/remove domains based on your industry and product categories.
# For furniture/home: manufacturer sites, interior design publications, standards bodies.
# Edit this list to match your company's domain.
# =============================================================================
ALLOWED_WEB_DOMAINS = [
    # Brand
    "thesleepcompany.in",

    # General reference
    "wikipedia.org",
    "britannica.com",

    # Sleep & health
    "sleepfoundation.org",
    "mayoclinic.org",
    "webmd.com",
    "healthline.com",
    "nhs.uk",

    # Certifications & standards
    "iso.org",
    "bis.gov.in",
    "astm.org",
    "oeko-tex.com",
    "certipur.us",

    # Industry & publications
    "architecturaldigest.com",
    "dezeen.com",
    "goodhousekeeping.com",

    # Reviews & comparisons
    "sleepopolis.com",
    "sleepadvisor.org",
    "tuck.com",

    # Competitors
    "wakefit.co",
    "sleepycat.in",
    "kurlon.com",
    "sleepwellproducts.com",
    "duroflexworld.com",

    # Materials & textiles
    "textileworld.com",
    "fibre2fashion.com",

    # Ergonomics & research
    "ergonomics.org.uk",
    "cdc.gov",

    # India-specific
    "consumerhelpline.gov.in"
]

HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type":  "application/json",
}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = "sales-chatbot"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("HF_API_KEY    :", "✅" if HF_API_KEY       else "❌ MISSING")
print("PINECONE_KEY  :", "✅" if PINECONE_API_KEY  else "❌ MISSING")
print("GROQ_API_KEY  :", "✅" if GROQ_API_KEY      else "❌ MISSING")

pc     = Pinecone(api_key=PINECONE_API_KEY)
index  = pc.Index(PINECONE_INDEX)
client = Groq(api_key=GROQ_API_KEY)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your frontend URL in production
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
    role:        str | None = None

class RatingRequest(BaseModel):
    rating: str   # "thumbs_up" or "thumbs_down"

class LoginRequest(BaseModel):
    email:    str
    password: str


# =============================================================================
# CHANGE 2 — QUERY PARSER (added query_type, removed route)
#
# query_type controls the top-level pipeline flow:
#   conversational → short-circuit, respond directly, no retrieval
#   informational  → LLM-only, no retrieval needed
#   retrieval      → full DB + conditional web pipeline
#
# doc_category and topic are only used when query_type = "retrieval".
# route has been removed — web vs DB logic is now handled inside
# retrieve_and_answer() based on DB result quality, not the parser.
# =============================================================================

def parse_query(user_query: str) -> dict:
    """
    Uses llama-3.1-8b-instant to extract structured intent from any query.
    Returns: { query_type, doc_category, topic }

    query_type:
        conversational = greetings, small talk, "how can you help", "what can you do"
        informational  = general knowledge question, no internal doc needed
        retrieval      = needs internal documents (products, SOPs, policies, training)

    doc_category mirrors the infer_doc_category() labels from ingest.py:
        product / policy / sop / training / pricing / faq / general

    topic is a short free-text subject used to build the Pinecone filter
    and embedding prefix. Only relevant when query_type = "retrieval".
    """
    prompt = f"""You are a query classifier for a 'The Sleep Company' assistant chatbot.
The assistant has access to internal documents including:
  - Product catalogs and specs
  - Company SOPs (Standard Operating Procedures)
  - HR and company policies
  - Training manuals and guides
  - Pricing documents

Given the user query below, return ONLY a valid JSON object with these fields:

  "query_type"   : one of ["conversational", "informational", "retrieval"]
                   conversational = greetings, small talk, "how can you help me", "what can you do"
                   informational  = general knowledge, no internal documents needed
                   retrieval      = needs internal company documents to answer

  "doc_category" : one of ["product", "policy", "sop", "training", "pricing", "faq", "general"]
                   Only relevant when query_type is "retrieval". Default to "general" otherwise.

  "topic"        : a short phrase (max 6 words) describing the specific subject.
                   Only relevant when query_type is "retrieval". Empty string otherwise.

Rules:
  - conversational ONLY IF: pure greeting ("hi", "hello", "thanks"), 
    small talk, or explicitly asks what you can do ("what can you help with")
  - informational ONLY IF: general world knowledge with zero connection 
    to sleep, sofas, furniture, mattresses, or company operations
  - retrieval FOR EVERYTHING ELSE — when in doubt, always choose retrieval
  - Short or vague queries about products ("sofa recommendations", 
    "mattress options", "best pillow") → ALWAYS retrieval, never informational
  - Default doc_category to "general" if unclear

User query: {user_query}"""

    try:
        response = client.chat.completions.create(
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

        print(f"[Parser] type={query_type} | category={doc_category} | topic={topic!r}")
        return {"query_type": query_type, "doc_category": doc_category, "topic": topic}

    except Exception as e:
        print(f"[Parser] Failed to parse intent ({e}) — using safe defaults")
        return {"query_type": "retrieval", "doc_category": "general", "topic": ""}


# =============================================================================
# CHANGE 3 — CONVERSATIONAL HANDLER (new)
#
# Short-circuits the pipeline for greetings and capability questions.
# No embedding, no Pinecone, no web search, no validation needed.
# Uses a warm, helpful tone and lists what the assistant can actually do.
# =============================================================================

def handle_conversational(user_query: str) -> str:
    """
    Handles greetings, small talk, and 'how can you help' queries directly.
    Bypasses all retrieval — pure LLM response.
    """
    prompt = f"""You are a helpful assistant for sales representatives and employees. The name of your company is 'The Sleep Company'.

The user has sent a conversational message — a greeting, small talk, or a question about what you can do.

Respond warmly and professionally. If they are asking what you can help with, mention:
- Product information, specs, and recommendations
- Company SOPs and procedures
- HR and company policies
- Training materials and guides
- Pricing information

Keep your response concise (2–4 sentences max). Do not make up capabilities you don't have.

User message: {user_query}"""

    try:
        response = client.chat.completions.create(
            model      = "llama-3.1-8b-instant",
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0.5,
            max_tokens = 150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Conversational] Error: {e}")
        return "Hello! I'm your company assistant. I can help you with product information, SOPs, policies, and training materials. What would you like to know?"


# =============================================================================
# STEP 2 — EMBEDDING  (unchanged, context prefix matches ingest.py exactly)
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
# STEP 3 — PINECONE RETRIEVAL  (unchanged — filter + fallback same as before)
#
# Returns matches WITH their scores so retrieve_and_answer() can judge quality.
# Only change: returns list of (metadata, score) tuples instead of just metadata.
# =============================================================================

def rewrite_query(user_query: str, memory_block: str) -> str:
    if not memory_block:
        return user_query
    prompt = f"""Rewrite the user's question as a fully self-contained search query.
Resolve any pronouns or references using the conversation history.
Return ONLY the rewritten query, nothing else.

Conversation history:
{memory_block}

User question: {user_query}
Rewritten query:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=60,
        )
        return response.choices[0].message.content.strip()
    except:
        return user_query

def retrieve_from_db(
    user_query:   str,
    doc_category: str,
    topic:        str,
    role:         str | None = None,
    top_k:        int = SOURCES_ACCESSED,
) -> list[dict]:
    """
    Returns list of dicts, each with all metadata fields plus an injected
    '_score' key so the caller can apply score-based quality checks.
    Pinecone data format is unchanged — same as ingest.py.
    """
    embed_text = build_query_embed_text(user_query, doc_category, topic)
    embedding  = get_embedding_with_retry(embed_text)

    pinecone_filter = {}
    if doc_category and doc_category != "general":
        pinecone_filter = {"doc_category": {"$eq": doc_category}}

    results = index.query(
        vector          = embedding,
        top_k           = top_k,
        include_metadata= True,
        filter          = pinecone_filter if pinecone_filter else None,
    )
    matches = results.get("matches", [])

    if len(matches) < 2 and pinecone_filter:
        print(f"[Retrieval] Filter '{doc_category}' too narrow — falling back to unfiltered")
        results = index.query(
            vector          = embedding,
            top_k           = top_k,
            include_metadata= True,
        )
        matches = results.get("matches", [])

    # Inject score into metadata so callers can threshold on it
    valid = []
    for m in matches:
        if m["metadata"].get("text"):
            chunk = dict(m["metadata"])
            chunk["_score"] = m.get("score", 0.0)
            valid.append(chunk)
    pinecone_filter = {}
    if doc_category and doc_category != "general":
        pinecone_filter["doc_category"] = {"$eq": doc_category}
    # Sales reps only see product/pricing/faq — not internal HR/SOP
    if role == "sales":
        pinecone_filter["doc_category"] = {
            "$in": ["product", "pricing", "faq", "general"]
        }
    print(f"[Retrieval] {len(valid)} chunks returned (filter={pinecone_filter or 'none'})")
    return valid


# =============================================================================
# CHANGE 4 — WEB SEARCH (domain allowlist + score filter + LLM relevance check)
#
# Three layers of filtering:
#   1. include_domains — Tavily only searches the allowlist above
#   2. score threshold  — drops low-confidence Tavily results (< WEB_SCORE_THRESHOLD)
#   3. LLM relevance check — asks a small model if the result is actually useful
#
# Max results capped at MAX_WEB_RESULTS to avoid noise flooding the context.
# =============================================================================

def is_web_result_relevant(user_query: str, result_content: str) -> bool:
    """
    Quick LLM check: is this web result actually relevant to the query?
    Uses the smallest/fastest model to keep latency low.
    Returns True if relevant, False if not.
    """
    prompt = f"""You are checking if a web search result is relevant to a user query.

User query: {user_query}

Web result content (first 400 chars):
{result_content[:400]}

Is this result relevant and useful for answering the query?
Reply ONLY with: yes OR no"""

    try:
        response = client.chat.completions.create(
            model      = "llama-3.1-8b-instant",
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0,
            max_tokens = 5,
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"[WebFilter] LLM check failed ({e}) — keeping result by default")
        return True


def search_web(query: str, max_chars: int = 800) -> list[dict]:
    """
    Searches web via Tavily with:
      - Domain allowlist (only trusted sources)
      - Score threshold (drop low-confidence results)
      - LLM relevance check (drop off-topic results)
      - Capped at MAX_WEB_RESULTS
    """
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        results = tavily.search(
            query          = query,
            max_results    = 5,                  # fetch more, filter down
            include_domains= ALLOWED_WEB_DOMAINS, # CHANGE: domain allowlist
        )

        filtered = []
        for r in results.get("results", []):

            # CHANGE: score threshold filter
            score = r.get("score", 0.0)
            if score < WEB_SCORE_THRESHOLD:
                print(f"[WebFilter] Dropped (score={score:.2f}): {r.get('url','')}")
                continue

            content = r.get("content", "")[:max_chars]

            # CHANGE: LLM relevance check
            if not is_web_result_relevant(query, content):
                print(f"[WebFilter] Dropped (irrelevant): {r.get('url','')}")
                continue

            filtered.append({
                "title":   r.get("title", "Web Source"),
                "url":     r.get("url", "#"),
                "content": content,
                "score":   score,
            })

            if len(filtered) >= MAX_WEB_RESULTS:  # CHANGE: cap results
                break

        print(f"[WebSearch] {len(filtered)} results kept after filtering")
        return filtered

    except Exception as e:
        print(f"[WebSearch] Failed: {e}")
        return []


# =============================================================================
# STEP 5 — CONTEXT BUILDER (unified — DB and web merged into one block)
#
# DB chunks appear first (priority position in context window).
# Web results appended after only if they were admitted by search_web().
# Source labels retained for traceability but not separated for the LLM.
# =============================================================================

def build_context(db_chunks: list[dict], web_results: list[dict]) -> str:
    parts = []

    # DB chunks first — higher priority position
    for chunk in db_chunks:
        source   = chunk.get("source", "internal document")
        heading  = chunk.get("heading", "")
        category = chunk.get("doc_category", "")
        text     = chunk.get("text", "")
        label    = f"[{category.upper()} — {source}] {heading}"
        parts.append(f"{label}\n{text}")

    # Web results after — only if admitted
    for w in web_results:
        parts.append(f"[WEB — {w['title']}]\n{w['content']}")

    return "\n\n---\n\n".join(parts)


# =============================================================================
# STEP 6 — CONVERSATION MEMORY (unchanged)
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
# STEP 7 — LLM CALLS (unchanged)
# =============================================================================

def query_llm(prompt: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2) -> str:
    try:
        response = client.chat.completions.create(
            model    = model,
            messages = [
                {
                    "role": "system",
                    "content": (
    "You are an internal company assistant for sales representatives and employees.\n\n"

    "Your role:\n"
    "- Help with product recommendations\n"
    "- Answer SOP, policy, and training-related questions\n"
    "- Assist sales reps in customer conversations\n\n"

    "CRITICAL RULES:\n"
    "- Use ONLY the provided context\n"
    "- Do NOT use outside knowledge\n"
    "- Do NOT hallucinate\n\n"

    "If information is missing:\n"
    "- Ask a clarifying question if it helps\n"
    "- OR say: 'I don't have that information in my documents. Please contact the relevant team.'\n\n"

    "QUESTION HANDLING:\n"
    "1. If query is vague:\n"
    "- Ask 1–2 clarifying questions OR give guided recommendation\n\n"

    "2. If query is specific:\n"
    "- Answer directly with exact details\n\n"

    "3. If recommending:\n"
    "- Suggest relevant products from context\n"
    "- Explain why briefly\n\n"

    "STYLE:\n"
    "- Professional\n"
    "- Concise\n"
    "- Point-wise\n"
    "- Plain text only\n\n"

    "COMPANY RULE:\n"
    "- Never portray company negatively\n"
),
                },
                {"role": "user", "content": prompt},
            ],
            temperature = temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM Error] {e}")
        return "I'm facing a temporary issue accessing the knowledge system. Please try again."


def validate_answer(user_query: str, answer: str, context: str) -> dict:
    prompt = f"""
You are a strict validator for an internal company assistant.

Check:

1. Grounding:
- Answer must ONLY use given context
- Any external info = INVALID

2. Relevance:
- Must directly address question

3. Completeness:
- Specific query → exact details required
- General query → recommendation OR clarification

4. Behavior:
- Vague query → ask clarification or guide
- Specific query → no unnecessary questions

5. Usefulness:
- Must help a sales rep or employee

6. Tone:
- Professional, concise, point-wise

7. Company safety:
- No negative statements

---

Context:
{context[:1500]}

Question:
{user_query}

Answer:
{answer}

---

Reply ONLY:

VALID: yes
FEEDBACK: <reason>

OR

VALID: no
FEEDBACK: <issue and fix>
"""

    response = client.chat.completions.create(
        model      = "llama-3.1-8b-instant",
        messages   = [{"role": "user", "content": prompt}],
        temperature= 0,
    )
    result   = response.choices[0].message.content.strip()
    valid    = "valid: yes" in result.lower()
    feedback = result.split("FEEDBACK:")[-1].strip() if "FEEDBACK:" in result else ""
    print(f"[Validator] valid={valid} | {feedback}")
    return {"valid": valid, "feedback": feedback}


def generate_followups(user_query: str, answer: str) -> list[str]:
    prompt = f"""Based on this Q&A from a company assistant, suggest 3 short follow-up questions.

Question: {user_query}
Answer: {answer}

Rules:
- Each question under 10 words
- Specific to the topic discussed (could be product, policy, SOP, training, etc.)
- Return ONLY a JSON array of 3 strings, no explanation
Example: ["What is the escalation process?", "Who approves this?", "Is there a form to fill?"]"""

    try:
        response = client.chat.completions.create(
            model      = "llama-3.1-8b-instant",
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0.4,
        )
        raw = response.choices[0].message.content.strip()
        suggestions = json.loads(raw)
        return suggestions if isinstance(suggestions, list) else []
    except Exception as e:
        print(f"[Followups] Error: {e}")
        return []


# =============================================================================
# CHANGE 5 — RETRIEVE AND ANSWER (unified DB + conditional web)
#
# Flow:
#   1. Always query Pinecone first
#   2. Count "strong" DB chunks (score >= SCORE_THRESHOLD)
#   3. If strong chunks >= DB_STRONG_THRESHOLD → skip web entirely
#   4. If strong chunks < DB_STRONG_THRESHOLD → run web search
#      Web results pass through domain allowlist + score filter + LLM check
#   5. Merge into single context block (DB first, web after)
#   6. Single LLM call on unified context
# =============================================================================

def retrieve_and_answer(
    user_query:   str,
    parsed:       dict,
    memory_block: str = "",
) -> tuple[str, dict, str]:
    """Returns (answer, sources_dict, context_string)"""

    doc_category = parsed["doc_category"]
    topic        = parsed["topic"]

    # --- Step 1: Always query DB first ---
    retrieval_query = rewrite_query(user_query, memory_block)
    db_chunks = retrieve_from_db(retrieval_query, doc_category, topic)
    # --- Step 2: Count strong DB chunks ---
    strong_chunks = [c for c in db_chunks if c.get("_score", 0.0) >= SCORE_THRESHOLD]
    print(f"[Pipeline] {len(strong_chunks)} strong DB chunks (threshold={SCORE_THRESHOLD})")

    # --- Step 3: Conditionally fetch web ---
    web_results = []
    if len(strong_chunks) < DB_STRONG_THRESHOLD:
        print(f"[Pipeline] DB weak — triggering web search")
        web_results = search_web(user_query)
    else:
        print(f"[Pipeline] DB strong — skipping web search")

    # --- Step 4: Build unified context (DB first = priority position) ---
    context = build_context(db_chunks, web_results)

    if not context.strip():
        sources = {"db_sources": [], "web_sources": []}
        return (
            "I don't have enough information in my documents to answer that. "
            "Please contact the relevant team.",
            sources,
            "",
        )

    memory_section = (
        f"--- CONVERSATION HISTORY ---\n{memory_block}\n----------------------------\n"
        if memory_block else ""
    )

    prompt = f"""{memory_section}
Use ONLY the following context to answer the question.
The context may include product information, company policies, SOPs, or training material.
If the answer is not present, say so clearly.

--- CONTEXT ---
{context}
---------------

Question: {user_query}
Answer:"""

    answer  = query_llm(prompt)
    sources = {
        "db_sources":  [c.get("text", "") for c in db_chunks],
        "web_sources": web_results,
    }

    return answer, sources, context


# =============================================================================
# CHANGE 6 — PROCESS QUERY (early exit for conversational + informational)
#
# query_type = conversational → handle_conversational(), return immediately
# query_type = informational  → LLM-only, no retrieval, no validation
# query_type = retrieval      → full pipeline as before
# =============================================================================

def process_query(user_query: str, memory_block: str = "") -> tuple[str, dict]:
    """
    Full pipeline with query_type-based routing:
      conversational → direct LLM response, no retrieval
      informational  → LLM-only, no retrieval
      retrieval      → DB + conditional web + validation + retry
    """
    parsed     = parse_query(user_query)
    query_type = parsed["query_type"]

    # --- Early exit: conversational ---
    if query_type == "conversational":
        print("[Pipeline] Conversational query — short-circuiting")
        answer = handle_conversational(user_query)
        return answer, {"db_sources": [], "web_sources": []}

    # --- Early exit: informational (no retrieval needed) ---
    if query_type == "informational":
        print("[Pipeline] Informational query — LLM only")
        prompt = f"Answer this general question concisely and professionally:\n\n{user_query}"
        answer = query_llm(prompt, model="llama-3.1-8b-instant", temperature=0.3)
        return answer, {"db_sources": [], "web_sources": []}

    # --- Full retrieval pipeline ---
    last_feedback = ""
    answer        = ""
    sources       = {"db_sources": [], "web_sources": []}

    for attempt in range(MAX_RETRIES):
        query_with_hint = user_query
        if attempt > 0 and last_feedback:
            query_with_hint = f"{user_query}\n\n[Fix this issue in your answer: {last_feedback}]"
            print(f"[Retry] Attempt {attempt + 1} — hint: {last_feedback}")

        answer, sources, context = retrieve_and_answer(query_with_hint, parsed, memory_block)

        if not context:
            return answer, sources

        validation = validate_answer(user_query, answer, context)

        if validation["valid"]:
            print(f"[Pipeline] ✅ Passed validation on attempt {attempt + 1}")
            return answer, sources

        last_feedback = validation["feedback"]

    print("[Pipeline] Max retries reached — returning best-effort answer")
    return answer, sources


# =============================================================================
# API ENDPOINTS (unchanged except web_sources in response)
# =============================================================================

@app.get("/")
def root():
    return {"status": "Company Assistant API is running"}


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

    # 3. Run pipeline
    answer, sources = process_query(request.query, memory_block)

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

    # 5. Generate follow-ups (skip for conversational)
    followups = []
    if sources["db_sources"] or sources["web_sources"]:
        followups = generate_followups(request.query, answer)

    return {
        "session_id"  : str(session.session_id),
        "message_id"  : str(message.message_id),
        "answer"      : answer,
        "db_sources"  : sources["db_sources"],
        "web_sources" : sources["web_sources"],   # renamed from internet_sources
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