# =============================================================================
# APP.PY — FASTAPI + GROQ + PINECONE + SUPABASE
# Generic RAG Chatbot Backend
#
# Works for any document type: products, SOPs, policies, training manuals.
# No domain knowledge is hardcoded — the LLM figures out intent at query time.
# =============================================================================

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

SOURCES_ACCESSED = 6    # top_k chunks pulled from Pinecone per query
MEMORY_TURNS     = 3    # how many past Q&A pairs to include in the LLM prompt
MAX_RETRIES      = 2    # validation retry attempts before returning best effort

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
# STEP 1 — QUERY PARSING  (LLM-based, no hardcoded lists)
#
# Instead of keyword-matching against fixed product/section lists,
# we ask a small fast LLM to extract:
#   - doc_category : what type of document is this query about?
#   - topic        : the specific subject within that document
#   - route        : db / web / both
#
# This works for products, SOPs, HR policies, training docs — anything.
# The LLM does the domain understanding so we don't have to.
# =============================================================================

def parse_query(user_query: str) -> dict:
    """
    Uses llama-3.1-8b-instant to extract structured intent from any query.
    Returns: { doc_category, topic, route }

    doc_category mirrors the infer_doc_category() labels from ingest.py:
        product / policy / sop / training / pricing / faq / general

    topic is a short free-text subject (e.g. "Valencia warranty",
    "maternity leave policy", "escalation procedure") used to build
    the Pinecone filter and embedding prefix.

    route controls retrieval source:
        db   = search Pinecone (internal documents)
        web  = search Tavily (external/live web)
        both = search both
    """
    prompt = f"""You are a query classifier for a company assistant chatbot.
The assistant has access to internal documents including:
  - Product catalogs and specs
  - Company SOPs (Standard Operating Procedures)
  - HR and company policies
  - Training manuals and guides
  - Pricing documents

Given the user query below, return ONLY a valid JSON object with these fields:

  "doc_category" : one of ["product", "policy", "sop", "training", "pricing", "faq", "general"]
  "topic"        : a short phrase (max 6 words) describing the specific subject
  "route"        : one of ["db", "web", "both"]
                   db   = answer from internal documents
                   web  = answer needs current/external web information
                   both = needs both internal docs and external info

Rules:
  - Default doc_category to "general" if unclear
  - Default route to "db" unless the query clearly needs live/external data
  - topic should be specific, not generic (e.g. "Valencia sofa warranty" not "warranty")
  - Return ONLY the JSON object, no explanation, no markdown

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

        # Validate and sanitise fields
        valid_categories = {"product", "policy", "sop", "training", "pricing", "faq", "general"}
        valid_routes     = {"db", "web", "both"}

        doc_category = parsed.get("doc_category", "general")
        topic        = parsed.get("topic", "")
        route        = parsed.get("route", "db")

        if doc_category not in valid_categories: doc_category = "general"
        if route        not in valid_routes:     route        = "db"
        if not isinstance(topic, str):           topic        = ""

        print(f"[Parser] category={doc_category} | topic={topic!r} | route={route}")
        return {"doc_category": doc_category, "topic": topic, "route": route}

    except Exception as e:
        # Fallback — never crash the pipeline on a parse failure
        print(f"[Parser] Failed to parse intent ({e}) — using safe defaults")
        return {"doc_category": "general", "topic": "", "route": "db"}


# =============================================================================
# STEP 2 — EMBEDDING  (with context prefix + retry)
#
# The prefix format MUST match build_embed_text() in ingest.py exactly:
#   "Category: <doc_category>. Heading: <topic/heading>.\n<text>"
#
# Matching the ingest prefix means query vectors land in the same
# region of embedding space as the stored document chunk vectors.
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
# STEP 3 — PINECONE RETRIEVAL  (filtered by doc_category, with fallback)
#
# We filter on doc_category when the parser is confident — this prevents
# a product query from pulling policy chunks and vice versa.
#
# If the filtered search returns < 2 results (filter too narrow),
# we automatically retry without the filter so we never return empty.
# =============================================================================

def retrieve_from_db(
    user_query:   str,
    doc_category: str,
    topic:        str,
    top_k:        int = SOURCES_ACCESSED,
) -> list[dict]:
    embed_text = build_query_embed_text(user_query, doc_category, topic)
    embedding  = get_embedding_with_retry(embed_text)

    # Only filter on doc_category when we have a specific category
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

    # Fallback: if filtered search returned too few results, go unfiltered
    if len(matches) < 2 and pinecone_filter:
        print(f"[Retrieval] Filter '{doc_category}' too narrow — falling back to unfiltered")
        results = index.query(
            vector          = embedding,
            top_k           = top_k,
            include_metadata= True,
        )
        matches = results.get("matches", [])

    valid = [m["metadata"] for m in matches if m["metadata"].get("text")]
    print(f"[Retrieval] {len(valid)} chunks returned (filter={pinecone_filter or 'none'})")
    return valid


# =============================================================================
# STEP 4 — WEB SEARCH  (Tavily)
# =============================================================================

def search_web(query: str, max_chars: int = 800) -> list[dict]:
    from tavily import TavilyClient
    tavily  = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily.search(query=query, max_results=3)

    return [
        {
            "title":   r.get("title", "Web Source"),
            "url":     r.get("url", "#"),
            "content": r.get("content", "")[:max_chars],
        }
        for r in results.get("results", [])
    ]


# =============================================================================
# STEP 5 — CONTEXT BUILDER
# Labels each chunk with its source document and heading so the LLM
# knows exactly where each fact came from.
# =============================================================================

def build_context(db_chunks: list[dict], web_results: list[dict]) -> str:
    parts = []

    for chunk in db_chunks:
        source   = chunk.get("source", "internal document")
        heading  = chunk.get("heading", "")
        category = chunk.get("doc_category", "")
        text     = chunk.get("text", "")
        label    = f"[{category.upper()} — {source}] {heading}"
        parts.append(f"{label}\n{text}")

    for w in web_results:
        parts.append(f"[WEB — {w['title']}]\n{w['content']}")

    return "\n\n---\n\n".join(parts)


# =============================================================================
# STEP 6 — CONVERSATION MEMORY
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
# STEP 7 — LLM CALLS
# =============================================================================

def query_llm(prompt: str, model: str = "llama-3.1-70b-versatile", temperature: float = 0.2) -> str:
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
# STEP 8 — CORE PIPELINE
# =============================================================================

def retrieve_and_answer(
    user_query:   str,
    parsed:       dict,
    memory_block: str = "",
) -> tuple[str, dict, str]:
    """Returns (answer, sources_dict, context_string)"""

    doc_category = parsed["doc_category"]
    topic        = parsed["topic"]
    route        = parsed["route"]

    db_chunks   = []
    web_results = []

    if route in ["db", "both"]:
        db_chunks = retrieve_from_db(user_query, doc_category, topic)

    if route in ["web", "both"]:
        web_results = search_web(user_query)

    context = build_context(db_chunks, web_results)

    if not context.strip():
        sources = {"db_sources": [], "internet_sources": []}
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
        "db_sources":       [c.get("text", "") for c in db_chunks],
        "internet_sources": web_results,
    }

    return answer, sources, context


def process_query(user_query: str, memory_block: str = "") -> tuple[str, dict]:
    """
    Full pipeline:
      1. Parse intent (LLM-based, no hardcoding)
      2. Retrieve from DB and/or web
      3. Generate answer
      4. Validate and retry if needed
    """
    parsed = parse_query(user_query)

    last_feedback = ""
    answer        = ""
    sources       = {"db_sources": [], "internet_sources": []}

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
# API ENDPOINTS
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

    # 5. Generate follow-ups
    followups = generate_followups(request.query, answer)

    return {
        "session_id"      : str(session.session_id),
        "message_id"      : str(message.message_id),
        "answer"          : answer,
        "db_sources"      : sources["db_sources"],
        "internet_sources": sources["internet_sources"],
        "followups"       : followups,
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