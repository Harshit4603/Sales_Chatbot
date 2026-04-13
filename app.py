# =============================================================================
# FINE-TUNED APP.PY: ADVANCED LLM ROUTER & MULTI-SOURCE RETRIEVAL
# WITH FASTAPI + SUPABASE LOGGING + CONVERSATION MEMORY
# =============================================================================
import requests
import numpy as np
import uuid
from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from database import get_db
from models import ChatSession, ChatMessage

# =========================
# CONFIGURATION
# =========================
OLLAMA_URL        = "http://localhost:11434/api/generate"
VECTOR_DB_PATH    = "chroma_db"
ROUTER_MODEL      = "mistral"
LLM_MODEL         = "mistral"

TOP_K_RETRIEVE    = 15  # Raw chunks pulled from vector DB before reranking
TOP_K_RERANK      = 3   # Chunks kept after reranking and sent to LLM
WEB_RESULTS_COUNT = 3   # Web results fetched

# How many previous Q&A pairs to include as memory
# 1 = only last question/answer (recommended to start)
# Raise to 2 or 3 if employees ask multi-step follow-ups frequently
MEMORY_TURNS      = 1

app = FastAPI()

_reranker_instance = None

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        print("Loading reranker model...")
        _reranker_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Reranker ready.\n")
    return _reranker_instance


# =========================
# RETRIEVER CLASS
# =========================
class Retriever:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        print("Loading vector database...")
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embedding_model
        )
        print("Retriever ready.\n")

    def search(self, query, top_k=TOP_K_RETRIEVE):
        results = self.vector_db.similarity_search_with_score(query, k=top_k)
        docs = []
        for doc, score in results:
            docs.append((doc.page_content, 1 - score, doc.metadata))
        return docs


# =========================
# LLM INTERACTION
# =========================
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_llm(prompt, model="llama-3.1-8b-instant", temperature=0.7):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful sales assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"LLM Error: {e}")
        return "Error connecting to LLM."
print(query_llm("Why is memory foam pillow good?"))
# =========================
# ROUTER LAYER
# =========================
def route_query(user_query):
    router_prompt = f"""
You are a high-precision query router for a retail store assistant.
Your goal is to determine the most relevant source(s) for a user's question.

CHOOSE EXACTLY ONE OPTION:
- "database": For questions about our specific products, store policies, inventory, catalog, or internal business details.
- "internet": For general knowledge, competitor comparisons, weather, news, or general technical definitions.
- "both": For complex queries requiring internal product data AND external context.

EXAMPLES:
Query: "What is your return policy for sofas?" -> database
Query: "Do you have any blue recliners in stock?" -> database
Query: "Who is the current CEO of Microsoft?" -> internet
Query: "Explain the concept of Mid-Century Modern design." -> internet
Query: "How does the durability of your Titan Sofa compare to standard industry recliners?" -> both
Query: "I have back pain; which of your chairs is best for lumbar support according to medical standards?" -> both

REAL USER QUERY:
"{user_query}"

OUTPUT ONLY THE WORD (database/internet/both):
"""
    response = query_llm(router_prompt, model=ROUTER_MODEL, temperature=0.0).strip().lower()
    if "database" in response: return "database"
    if "internet" in response: return "internet"
    if "both" in response: return "both"
    return "both"


# =========================
# CONTEXT REFINEMENT
# =========================
def refine_context(query, context):
    prompt = f"""
You are a context extraction specialist.
Given a query and raw context from multiple sources, extract ONLY the facts relevant to answering the query.

Question: {query}
Raw Context:
{context}

Relevant Facts:
"""
    return query_llm(prompt, temperature=0.2)


# =========================
# CONVERSATION MEMORY BUILDER
# =========================
def build_memory_block(session_id: str, db: Session) -> str:
    """
    Fetches the last MEMORY_TURNS Q&A pairs from the DB for this session
    and formats them as a conversation history block for the LLM.
    Returns an empty string if this is the first message.
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
_retriever_instance = None

def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance

def rerank(query, docs, top_k=TOP_K_RERANK):
    reranker = get_reranker()
    pairs = [(query, doc[0]) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for (doc, score) in ranked[:top_k]]

def process_query(user_query, memory_block: str = ""):
    route = route_query(user_query)
    print(f"[*] Route Decision: {route.upper()}")

    context_segments = []
    db_sources       = set()
    internet_sources = []

    # --- DATABASE RETRIEVAL ---
    if route in ["database", "both"]:
        print("[*] Accessing Internal Database...")
        retriever = get_retriever()
        results   = retriever.search(user_query, top_k=TOP_K_RETRIEVE)
        reranked  = rerank(user_query, results, top_k=TOP_K_RERANK)

        print(f"[*] Chunks retrieved: {len(results)} | Chunks sent to LLM after rerank: {len(reranked)}")

        db_text = ""
        for i, (doc, score, metadata) in enumerate(reranked):
            db_text += f"[Internal Chunk {i+1} | Relevance: {score:.2f}]: {doc}\n"
            db_sources.add(metadata.get("source", "Store Database"))

        if db_text:
            context_segments.append(f"--- INTERNAL STORE KNOWLEDGE ---\n{db_text}")

    # --- INTERNET RETRIEVAL ---
    if route in ["internet", "both"]:
        print("[*] Searching the Internet...")
        try:
            search_wrapper     = DuckDuckGoSearchAPIWrapper()
            web_search_results = search_wrapper.results(user_query, max_results=WEB_RESULTS_COUNT)

            print(f"[*] Web results fetched: {len(web_search_results)}")

            web_text = ""
            for i, res in enumerate(web_search_results):
                snippet = res.get("snippet", "")
                link    = res.get("link", "")
                title   = res.get("title", "Web Result")

                web_text += f"[Web Result {i+1} ({title})]: {snippet} (Source: {link})\n\n"
                internet_sources.append({"title": title, "url": link})

            if web_text:
                context_segments.append(f"--- EXTERNAL INTERNET DATA ---\n{web_text}")

        except Exception as e:
            print(f"Web Search Error: {e}")

    if not context_segments:
        return "I'm sorry, I couldn't find any information to answer that question.", {
            "db_sources": [], "internet_sources": []
        }

    full_context    = "\n\n".join(context_segments)
    refined_context = refine_context(user_query, full_context)

    # --- Build final prompt with memory injected ---
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
You are a helpful assistant for a retail store.
Answer the user's question based ONLY on the provided context.

RULES:
- If sources conflict, prioritize INTERNAL STORE KNOWLEDGE.
- If the answer is not in the context, say you do not know.
- Cite your source generally (e.g., "According to our database..." or "Based on web search...").
- If using web information, mention the specific website URL in your answer.
- If the question is a follow-up, use the conversation history to resolve references.
{memory_section}
Context:
{refined_context}

User Question: {user_query}
Answer:
"""
    answer = query_llm(final_prompt)

    sources = {
        "db_sources"      : sorted(list(db_sources)),
        "internet_sources": internet_sources
    }

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
        session = db.query(ChatSession).filter(
            ChatSession.session_id == request.session_id
        ).first()
    else:
        session = None

    if not session:
        session = ChatSession(employee_id=request.employee_id)
        db.add(session)
        db.commit()
        db.refresh(session)

    # 2. Build memory from previous messages in this session
    memory_block = build_memory_block(str(session.session_id), db)

    # 3. Process query with memory injected
    answer, sources = process_query(request.query, memory_block)

    # 4. Log to Supabase
    message = ChatMessage(
        session_id  = session.session_id,
        employee_id = request.employee_id,
        query       = request.query,
        answer      = answer,
    )
    db.add(message)
    session.last_active_at = datetime.utcnow()
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
        return {"error": "Message not found"}

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


# =========================
# TERMINAL INTERFACE
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
