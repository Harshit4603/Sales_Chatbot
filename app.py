# =============================================================================
# LIGHTWEIGHT APP.PY: FASTAPI + GROQ + SUPABASE LOGGING
# =============================================================================
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from groq import Groq
import requests
import json
from pydantic import BaseModel
from database import get_db
from models import ChatSession, ChatMessage, get_ist
from models import Employee  # make sure Employee model exists

# =========================
# CONFIGURATION
# =========================

SOURCES_ACCESSED = 10  # how many retrieved chunks to include in LLM context (DB + Web combined)

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5"

HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

from pinecone import Pinecone

# Initialize Pinecone (add this near top with config)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("sales-chatbot")  # your index name

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for now (later restrict to your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq Client Initialization
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[!] Warning: GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

# How many previous Q&A pairs to include as memory
MEMORY_TURNS = 1

def route_query(user_query: str) -> str:
    """Returns: 'db', 'web', or 'both'"""
    prompt = f"""You are a query classifier for a retail store assistant.

Classify the query into exactly one category:

db   = product catalog, pricing, stock levels, store hours, policies, order history
web  = competitor prices, news, real-time market data, external brand info  
both = requires internal product data AND external/current information

Rules:
- If in doubt, choose "db"
- Output ONLY the single word: db, web, or both
- No punctuation, no explanation

Query: {user_query}"""

    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=5  # hard cap
)
    decision = response.choices[0].message.content.strip().lower()
    print(f"[Router] Decision: {decision}")
    return decision if decision in ["db", "web", "both"] else "db"

# =========================
# LLM INTERACTION
# =========================
def query_llm(prompt, model="llama-3.1-8b-instant", temperature=0.2):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a retail store assistant Chatbot. "
                    "Answer using the context provided under 'Context:'. "
                    "Do NOT use your general knowledge."
                    "Return plain text only — no HTML, CSS, or markdown."
                    "Use a friendly and professional tone."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Error connecting to LLM."
    
# =========================
# CONVERSATION MEMORY BUILDER
# =========================
def build_memory_block(session_id: str, db: Session) -> str:
    """
    Fetches the last MEMORY_TURNS Q&A pairs from the DB for this session
    and formats them as a conversation history block for the LLM.
    """
    if not session_id:
        return ""

    past_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(MEMORY_TURNS)
        .all()
    )

    if not past_messages:
        return ""

    # Reverse so oldest comes first
    past_messages = list(reversed(past_messages))

    memory_lines = []
    for msg in past_messages:
        memory_lines.append(f"Previous Question: {msg.query}")
        memory_lines.append(f"Previous Answer: {msg.answer}")

    memory_block = "\n".join(memory_lines)
    print(f"[*] Memory loaded: {len(past_messages)} previous turn(s)")
    return memory_block

def get_embedding(text):
    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json={"inputs": text}
    )

    result = response.json()
    print(f"[*] HF response type: {type(result)}")
    print(f"[*] HF response first element type: {type(result[0])}")

    # ✅ Handle all 3 possible response shapes
    if isinstance(result, list) and isinstance(result[0], float):
        # Shape: [0.01, 0.04, ...] → flat list directly
        print(f"[*] Embedding shape: flat list, dim={len(result)}")
        return result
    elif isinstance(result, list) and isinstance(result[0], list) and isinstance(result[0][0], float):
        # Shape: [[0.01, 0.04, ...]] → single nested list
        print(f"[*] Embedding shape: nested list, dim={len(result[0])}")
        return result[0]
    elif isinstance(result, list) and isinstance(result[0], list) and isinstance(result[0][0], list):
        # Shape: [[[0.01, 0.04, ...]]] → double nested list
        print(f"[*] Embedding shape: double nested list, dim={len(result[0][0])}")
        return result[0][0]
    else:
        print(f"[!] Unknown embedding format: {str(result)[:200]}")
        raise ValueError("Unexpected embedding response format from HuggingFace")
    
# --- Update your search_web function in app.py ---
def search_web(query: str, max_chars_per_result=800) -> list:
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily.search(query=query, max_results=3)
    
    # FIX: Return a list of dicts instead of strings
    formatted_results = []
    for r in results.get("results", []):
        formatted_results.append({
            "title": r.get("title", "Web Source"),
            "url": r.get("url", "#"),
            "content": r.get("content", "")[:max_chars_per_result]  # hard cap
        })

    return formatted_results

# --- Update retrieve_and_answer to always return 3 items ---
def retrieve_and_answer(user_query: str, route: str, memory_block: str = ""):
    context_chunks = []
    sources = {"db_sources": [], "internet_sources": []}

    if route in ["db", "both"]:
        embedding = get_embedding(user_query)
        results = index.query(vector=embedding, top_k=SOURCES_ACCESSED, include_metadata=True)
        db_chunks = [m["metadata"]["text"] for m in results["matches"] if m["metadata"].get("text")]
        context_chunks.extend(db_chunks)
        sources["db_sources"] = db_chunks

    if route in ["web", "both"]:
        web_results = search_web(user_query)
        # Add only content to context, but keep full dicts for sources
        context_chunks.extend([r["content"] for r in web_results])
        sources["internet_sources"] = web_results

    if not context_chunks:
        # FIX: Always return 3 items (answer, sources, context_chunks)
        return "I don't have enough information to answer that.", sources, []

    context = "\n\n".join(context_chunks)
    memory_section = f"--- CONVERSATION HISTORY ---\n{memory_block}\n----------------------------\n" if memory_block else ""

    prompt = f"""{memory_section}
Use ONLY the following context to answer. Do not use outside knowledge.

--- CONTEXT ---
{context}
---------------

Question: {user_query}
Answer:"""

    answer = query_llm(prompt)
    return answer, sources, context_chunks

def validate_answer(user_query: str, answer: str, context_chunks: list) -> dict:
    """Returns: {'valid': bool, 'feedback': str}"""
    context = "\n\n".join(context_chunks)

    prompt = f"""Validate this retail assistant answer. Reply ONLY:
VALID: yes
FEEDBACK: <one sentence>

OR

VALID: no  
FEEDBACK: <specific problem>

Context: {context[:1500]}  ← cap this too
Question: {user_query}
Answer: {answer}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip()
    valid = "VALID: yes" in result.lower()
    feedback = result.split("FEEDBACK:")[-1].strip() if "FEEDBACK:" in result else ""
    print(f"[Validator] Valid: {valid} | Feedback: {feedback}")
    return {"valid": valid, "feedback": feedback}

def generate_followups(user_query: str, answer: str) -> list[str]:
    prompt = f"""You are a retail store assistant. Based on the conversation below, suggest 3 short follow-up questions the user might want to ask next.

Question: {user_query}
Answer: {answer}

Rules:
- Each suggestion must be under 10 words
- Make them specific to the topic discussed
- Return ONLY a JSON array of 3 strings, nothing else
Example: ["What are the return policy details?", "Do you offer EMI options?", "Is this available in other colors?"]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        raw = response.choices[0].message.content.strip()
        suggestions = json.loads(raw)
        return suggestions if isinstance(suggestions, list) else []
    except Exception as e:
        print(f"[Followups] Error: {e}")
        return []
# =========================
# CORE PIPELINE
# =========================
MAX_RETRIES = 2

def process_query(user_query: str, memory_block: str = ""):
    route = route_query(user_query)

    for attempt in range(MAX_RETRIES):
        retry_hint = ""
        if attempt > 0:
            retry_hint = f"\n\nPrevious attempt was rejected. Fix this issue: {last_feedback}"
            print(f"[Retry] Attempt {attempt + 1}, feedback: {last_feedback}")

        answer, sources, context_chunks = retrieve_and_answer(
            user_query + retry_hint, route, memory_block
        )

        if not context_chunks:
            return answer, sources

        validation = validate_answer(user_query, answer, context_chunks)

        if validation["valid"]:
            print(f"[Pipeline] Answer passed validation on attempt {attempt + 1}")
            return answer, sources

        last_feedback = validation["feedback"]

    # Return best effort after retries
    print("[Pipeline] Max retries reached, returning last answer")
    return answer, sources

# =========================
# REQUEST MODELS
# =========================
class ChatRequest(BaseModel):
    employee_id: str
    session_id: str | None = None
    query: str
    role: str | None = None  # add this

class RatingRequest(BaseModel):
    rating: str  # "thumbs_up" or "thumbs_down"

# =========================
# API ENDPOINTS
# =========================

@app.get("/")
def root():
    return {"status": "Store Assistant API is running"}

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    employee = db.query(Employee).filter(Employee.email == request.email).first()
    
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    if employee.password_hash != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "employee_id": employee.employee_id,
        "name": employee.name,
        "role": employee.role
    }

@app.post("/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    # 1. Get or create session
    if request.session_id:
        try:
            session = db.query(ChatSession).filter(
                ChatSession.session_id == request.session_id
            ).first()
        except Exception:
            session = None
    else:
        session = None

    if not session:
        session = ChatSession(employee_id=request.employee_id)
        db.add(session)
        db.commit()
        db.refresh(session)

    # 2. Build memory from previous messages in this session
    memory_block = build_memory_block(str(session.session_id), db)

    # 3. Process query
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

    # After: answer, sources = process_query(...)
    followups = generate_followups(request.query, answer)   # ← add this

    return {
        "session_id"      : str(session.session_id),
        "message_id"      : str(message.message_id),
        "answer"          : answer,
        "db_sources"      : sources["db_sources"],
        "internet_sources": sources["internet_sources"],
        "followups"       : followups,                      # ← add this
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
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.timestamp).all()

    return {
        "session_id": session_id,
        "messages": [
            {
                "message_id": str(m.message_id),
                "query"     : m.query,
                "answer"    : m.answer,
                "rating"    : m.rating,
                "timestamp" : str(m.timestamp)
            }
            for m in messages
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
