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

from database import get_db
from models import ChatSession, ChatMessage, get_ist

# =========================
# CONFIGURATION
# =========================
load_dotenv()

app = FastAPI()

# Groq Client Initialization
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[!] Warning: GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

# How many previous Q&A pairs to include as memory
MEMORY_TURNS = 1

# =========================
# LLM INTERACTION
# =========================
def query_llm(prompt, model="llama-3.1-8b-instant", temperature=0.7):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful retail store assistant. Answer questions concisely and professionally. IMPORTANT: Return ONLY the plain text answer. Do NOT include any HTML, CSS, or UI tags in your response."},
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

# =========================
# CORE PIPELINE
# =========================
def process_query(user_query, memory_block: str = ""):
    # Build final prompt with memory injected
    memory_section = ""
    if memory_block:
        memory_section = f"""
--- CONVERSATION HISTORY ---
{memory_block}
(Use this history to understand follow-up questions. If the user references
"it", "that", "the one you mentioned" etc., resolve them using the history above.)
----------------------------
"""

    final_prompt = f"""
{memory_section}
User Question: {user_query}
Answer:
"""
    answer = query_llm(final_prompt)
    sources = {"db_sources": [], "internet_sources": []}
    return answer, sources

# =========================
# REQUEST MODELS
# =========================
class ChatRequest(BaseModel):
    employee_id: str
    session_id: str | None = None
    query: str

class RatingRequest(BaseModel):
    rating: str  # "thumbs_up" or "thumbs_down"

# =========================
# API ENDPOINTS
# =========================

@app.get("/")
def root():
    return {"status": "Store Assistant API is running"}

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

    return {
        "session_id"      : str(session.session_id),
        "message_id"      : str(message.message_id),
        "answer"          : answer,
        "db_sources"      : sources["db_sources"],
        "internet_sources": sources["internet_sources"],
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
