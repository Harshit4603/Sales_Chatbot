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

from database import get_db
from models import ChatSession, ChatMessage, get_ist

# =========================
# CONFIGURATION
# =========================
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
def query_llm(prompt, model="llama-3.1-8b-instant", temperature=0.2):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a retail store assistant. "
                    "Answer ONLY using the context provided under 'Context:'. "
                    "If the context does not contain enough information to answer, "
                    "say: 'I don't have that information in my database.' "
                    "Do NOT use your general knowledge. "
                    "Return plain text only — no HTML, CSS, or markdown."
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
# =========================
# CORE PIPELINE
# =========================
def process_query(user_query, memory_block: str = ""):
    
    # 🔹 Step 1: Convert query to embedding
    query_embedding = get_embedding(user_query)
    print(f"[*] Embedding dimension: {len(query_embedding)}")

    # 🔹 Step 2: Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    # ✅ Debug: How many matches returned
    matches = results["matches"]
    print(f"[*] Pinecone matches returned: {len(matches)}")

    if not matches:
        print("[!] WARNING: No matches found in Pinecone!")
        return "I don't have that information in my database.", {"db_sources": [], "internet_sources": []}

    # ✅ Debug: Print each match score and preview
    for i, match in enumerate(matches):
        score = match.get("score", "N/A")
        text_preview = match["metadata"].get("text", "")[:100]
        print(f"  [{i+1}] Score: {score:.4f} | Text preview: {text_preview}...")

    # 🔹 Step 3: Extract context
    context_chunks = []
    for match in matches:
        text = match["metadata"].get("text", "").strip()
        if text:
            context_chunks.append(text)

    print(f"[*] Non-empty context chunks: {len(context_chunks)}")

    if not context_chunks:
        print("[!] WARNING: Matches found but all have empty text metadata!")
        return "I don't have that information in my database.", {"db_sources": [], "internet_sources": []}

    context = "\n\n".join(context_chunks)

    # 🔹 Step 4: Build prompt
    memory_section = ""
    if memory_block:
        memory_section = f"""
--- CONVERSATION HISTORY ---
{memory_block}
----------------------------
"""

    final_prompt = f"""Use ONLY the following context to answer the question. Do not use outside knowledge.

{memory_section}

--- CONTEXT FROM INTERNAL DATABASE ---
{context}
--------------------------------------

User Question: {user_query}

Answer (based only on the context above):"""

    # 🔹 Step 5: Call LLM
    print(f"[*] Sending prompt to LLM...")
    answer = query_llm(final_prompt)
    print(f"[*] LLM Response: {answer[:100]}...")

    return answer, {"db_sources": context_chunks, "internet_sources": []}

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
