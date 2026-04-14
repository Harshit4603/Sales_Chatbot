import streamlit as st
import requests
import time
import base64
import datetime

# =========================
# CONFIGURATION
# =========================
BACKEND_URL = "https://web-production-498498.up.railway.app"   # Your FastAPI URL
EMPLOYEE_ID = "EMP001"                   # Replace with real login later

# ── helpers ──────────────────────────────────────────────────────────────────

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def init_backend():
    """Warm up backend by hitting the root endpoint."""
    try:
        requests.get(f"{BACKEND_URL}/", timeout=5)
    except Exception:
        pass

def send_message(query: str) -> dict:
    """
    POST /chat to FastAPI.
    Automatically sends session_id on follow-up messages.
    Returns full response dict.
    """
    payload = {
        "employee_id": EMPLOYEE_ID,
        "query"      : query,
        "session_id" : st.session_state.get("session_id")  # None on first message
    }
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=None)
        response.raise_for_status()
        data = response.json()

        # Save session_id after first message — reused for all follow-ups
        if not st.session_state.get("session_id"):
            st.session_state.session_id = data.get("session_id")

        return data

    except Exception as e:
        return {"answer": f"Error connecting to server: {e}", "db_sources": [], "internet_sources": [], "message_id": None}

def rate_message(message_id: str, rating: str):
    """PATCH /chat/{message_id}/rate"""
    if not message_id:
        return
    try:
        requests.patch(
            f"{BACKEND_URL}/chat/{message_id}/rate",
            json={"rating": rating},
            timeout=10
        )
    except Exception:
        pass

# ── Google Fonts + global CSS ─────────────────────────────────────────────────

GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── reset Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: transparent !important; }

/* ── CSS variables ── */
:root {
    --navy:   #0A0F1E;
    --navy2:  #111827;
    --accent: #63B3ED;
    --accent2:#3B82F6;
    --text:   #E8F4FF;
    --muted:  #94A3B8;
    --green:  #4CAF7D;
    --red:    #EF5350;
    --bubble-bot: #FFFFFF;
    --bubble-user:#111827;
    --bg-page:#F0F2F8;
    --border: rgba(0,0,0,0.08);
    --font-serif: 'Cormorant Garamond', Georgia, serif;
    --font-sans:  'DM Sans', system-ui, sans-serif;
}

body { font-family: var(--font-sans); }
"""

# ── Loading Screen CSS ────────────────────────────────────────────────────────

LOADING_CSS = """
<style>
""" + GLOBAL_CSS + """

.stApp { background: var(--navy) !important; }

.bg-grid {
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(99,179,237,0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,179,237,0.045) 1px, transparent 1px);
    background-size: 30px 30px;
    pointer-events: none; z-index: 0;
}
.bg-glow {
    position: fixed; width: 560px; height: 560px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56,182,255,0.11) 0%, transparent 68%);
    top: 50%; left: 50%; transform: translate(-50%, -54%);
    pointer-events: none; z-index: 0;
    animation: pulse-glow 3.5s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%,100% { opacity:.6; transform:translate(-50%,-54%) scale(1);   }
    50%      { opacity:1;  transform:translate(-50%,-54%) scale(1.12);}
}
.loading-wrapper {
    position: fixed; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; z-index: 10;
}
.logo-ring {
    width: 88px; height: 88px; border-radius: 50%;
    border: 1.5px solid rgba(99,179,237,0.35);
    display: flex; align-items: center; justify-content: center;
    position: relative; margin-bottom: 34px;
    animation: spin-cw 9s linear infinite;
}
.logo-ring::before {
    content: ''; position: absolute;
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--accent); top: -5px; left: 50%;
    transform: translateX(-50%); box-shadow: 0 0 12px var(--accent);
}
.logo-inner {
    width: 58px; height: 58px; border-radius: 50%;
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.22);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; animation: spin-cw 9s linear infinite reverse;
}
@keyframes spin-cw { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.loading-name {
    font-family: var(--font-serif); font-size: 3.4rem; font-weight: 600;
    color: var(--text); letter-spacing: .06em; text-align: center;
    opacity: 0; transform: translateY(18px);
    animation: fade-up 1s ease .2s forwards;
}
.loading-tag {
    font-family: var(--font-sans); font-size: .72rem; font-weight: 300;
    letter-spacing: .22em; text-transform: uppercase; color: var(--accent);
    margin-top: 10px; text-align: center;
    opacity: 0; animation: fade-up 1s ease .55s forwards;
}
.dot-row { display: flex; gap: 7px; margin-top: 28px; justify-content: center; }
.dot-row span {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); opacity: .3;
    animation: dot-p 1.4s ease-in-out infinite;
}
.dot-row span:nth-child(2) { animation-delay:.2s; }
.dot-row span:nth-child(3) { animation-delay:.4s; }
@keyframes dot-p {
    0%,100% { opacity:.3; transform:scale(1);   }
    50%      { opacity:1;  transform:scale(1.45);}
}
@keyframes fade-up { to { opacity:1; transform:translateY(0); } }
</style>
"""

# ── Chat Screen CSS ───────────────────────────────────────────────────────────

CHAT_CSS = """
<style>
""" + GLOBAL_CSS + """

.stApp { background: var(--bg-page) !important; }

.sa-header {
    background: var(--navy); padding: 18px 28px 14px;
    display: flex; align-items: center; gap: 14px;
    position: sticky; top: 0; z-index: 100;
}
.sa-avatar {
    width: 40px; height: 40px; border-radius: 50%;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.28);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.sa-header-name {
    font-family: var(--font-serif); font-size: 1.35rem;
    font-weight: 600; color: var(--text); letter-spacing: .04em;
}
.sa-status {
    font-family: var(--font-sans); font-size: .7rem;
    font-weight: 300; color: var(--green);
    display: flex; align-items: center; gap: 5px; margin-top: 2px;
}
.sa-status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); animation: dot-p 2s ease-in-out infinite;
}
.chat-area { max-width: 780px; margin: 0 auto; padding: 28px 20px 100px; }

.msg-row-bot {
    display: flex; gap: 10px; align-items: flex-end;
    margin-bottom: 6px; animation: slide-left .3s ease;
}
.msg-row-user {
    display: flex; justify-content: flex-end;
    margin-bottom: 6px; animation: slide-right .3s ease;
}
@keyframes slide-left  { from { opacity:0; transform:translateX(-12px);} to {opacity:1;transform:translateX(0);}}
@keyframes slide-right { from { opacity:0; transform:translateX(12px); } to {opacity:1;transform:translateX(0);}}

.bot-mini-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    background: var(--navy); border: 1px solid rgba(99,179,237,0.28);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; flex-shrink: 0;
}
.bubble-bot-wrap { max-width: 72%; }
.bubble-bot {
    background: var(--bubble-bot); border: .5px solid var(--border);
    border-radius: 18px 18px 18px 4px; padding: 12px 16px;
    font-family: var(--font-sans); font-size: .88rem;
    line-height: 1.65; color: #1A2035;
}
.bubble-user {
    background: var(--navy); border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; font-family: var(--font-sans);
    font-size: .88rem; line-height: 1.65; color: var(--text); max-width: 72%;
}
.bubble-meta {
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 5px; padding: 0 2px;
}
.bubble-time { font-family: var(--font-sans); font-size: .65rem; color: var(--muted); }

/* sources badge */
.sources-row { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 5px; }
.source-badge {
    font-family: var(--font-sans); font-size: .62rem;
    background: #EFF6FF; border: .5px solid #BFDBFE;
    color: #1D4ED8; border-radius: 20px; padding: 2px 9px;
    text-decoration: none; white-space: nowrap;
}
.source-badge.db { background:#F0FDF4; border-color:#BBF7D0; color:#15803D; }

.rate-btns { display: flex; gap: 5px; }
.rate-btn {
    width: 26px; height: 26px; border-radius: 50%;
    border: .5px solid rgba(0,0,0,0.13); background: white;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 12px;
    transition: transform .15s, background .15s; padding: 0;
}
.rate-btn:hover { transform: scale(1.18); background:#f0f4ff; }
.rate-btn.liked    { background:#e8f5e9; border-color: var(--green); }
.rate-btn.disliked { background:#fff3f3; border-color: var(--red);   }

.stChatInput > div {
    border-radius: 28px !important;
    border: 1px solid rgba(10,15,30,0.15) !important;
    background: white !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    font-family: var(--font-sans) !important;
}
</style>
"""

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sales Assistant", page_icon="◈", layout="centered")

# ── session defaults ──────────────────────────────────────────────────────────

if "initialized"  not in st.session_state: st.session_state.initialized  = False
if "messages"     not in st.session_state: st.session_state.messages     = []
if "ratings"      not in st.session_state: st.session_state.ratings      = {}
if "session_id"   not in st.session_state: st.session_state.session_id   = None  # ← FastAPI session
if "message_ids"  not in st.session_state: st.session_state.message_ids  = {}    # {msg_index: message_id}

# ── LOADING SCREEN ────────────────────────────────────────────────────────────

if not st.session_state.initialized:
    st.markdown(LOADING_CSS, unsafe_allow_html=True)

    placeholder      = st.empty()
    type_placeholder = st.empty()

    with placeholder.container():
        st.markdown("""
        <div class="bg-grid"></div>
        <div class="bg-glow"></div>
        <div class="loading-wrapper">
            <div class="logo-ring"><div class="logo-inner">◈</div></div>
            <div class="loading-name">SalesAssist</div>
            <div class="loading-tag">Your intelligent sales companion</div>
        </div>
        """, unsafe_allow_html=True)

    time.sleep(1.0)

    sentence = "Connecting to your assistant..."
    typed = ""
    for char in sentence:
        typed += char
        type_placeholder.markdown(
            f'<div style="position:fixed;top:62%;left:0;width:100vw;text-align:center;'
            f'font-family:\'DM Sans\',sans-serif;font-size:.98rem;font-weight:300;'
            f'color:#94A3B8;letter-spacing:.02em;z-index:20;">{typed}▍</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.055)

    time.sleep(0.5)

    with st.spinner(""):
        init_backend()

    st.session_state.initialized = True
    placeholder.empty()
    type_placeholder.empty()
    st.rerun()

# ── CHAT SCREEN ───────────────────────────────────────────────────────────────

if st.session_state.initialized:

    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="sa-header">
        <div class="sa-avatar">◈</div>
        <div>
            <div class="sa-header-name">SalesAssist</div>
            <div class="sa-status">
                <div class="sa-status-dot"></div>
                Online · Ready to help
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-area">', unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        ts      = msg.get("time", "")
        role    = msg["role"]
        content = msg["content"]

        if role == "assistant":
            rating       = st.session_state.ratings.get(i)
            liked_cls    = "liked"    if rating == "up"   else ""
            disliked_cls = "disliked" if rating == "down" else ""

            # Build sources HTML
            db_sources       = msg.get("db_sources", [])
            internet_sources = msg.get("internet_sources", [])
            sources_html     = ""

            if db_sources or internet_sources:
                sources_html = '<div class="sources-row">'
                for src in db_sources:
                    sources_html += f'<span class="source-badge db">📄 {src}</span>'
                for src in internet_sources:
                    title = src.get("title", "Web")
                    url   = src.get("url", "#")
                    sources_html += f'<a class="source-badge" href="{url}" target="_blank">🌐 {title}</a>'
                sources_html += '</div>'

            st.markdown(f"""
            <div class="msg-row-bot">
                <div class="bot-mini-avatar">◈</div>
                <div class="bubble-bot-wrap">
                    <div class="bubble-bot">{content}</div>
                    {sources_html}
                    <div class="bubble-meta">
                        <span class="bubble-time">{ts}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Rating buttons
            col_gap, col_up, col_dn, col_rest = st.columns([6, 0.45, 0.45, 3])
            with col_up:
                if st.button("👍", key=f"up_{i}", help="Helpful"):
                    st.session_state.ratings[i] = "up"
                    rate_message(st.session_state.message_ids.get(i), "thumbs_up")
                    st.rerun()
            with col_dn:
                if st.button("👎", key=f"dn_{i}", help="Not helpful"):
                    st.session_state.ratings[i] = "down"
                    rate_message(st.session_state.message_ids.get(i), "thumbs_down")
                    st.rerun()

        else:
            st.markdown(f"""
            <div class="msg-row-user">
                <div class="bubble-user">{content}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Input ──
    if prompt := st.chat_input("Ask me anything about products, pricing, or deals..."):
        now = datetime.datetime.now().strftime("%I:%M %p")

        st.session_state.messages.append({
            "role": "user", "content": prompt, "time": now,
        })
        st.rerun()

# ── Generate response after rerun ────────────────────────────────────────────

if st.session_state.initialized and len(st.session_state.messages) > 0:
    last_msg = st.session_state.messages[-1]

    if last_msg["role"] == "user":
        with st.spinner("Analyzing request..."):
            data = send_message(last_msg["content"])

        answer           = data.get("answer", "Sorry, something went wrong.")
        db_sources       = data.get("db_sources", [])
        internet_sources = data.get("internet_sources", [])
        message_id       = data.get("message_id")

        idx = len(st.session_state.messages)

        st.session_state.messages.append({
            "role"            : "assistant",
            "content"         : answer,
            "time"            : datetime.datetime.now().strftime("%I:%M %p"),
            "db_sources"      : db_sources,
            "internet_sources": internet_sources,
        })

        # Store message_id so rating buttons can call /rate endpoint
        st.session_state.message_ids[idx] = message_id
        st.session_state.ratings[idx]     = None
        st.rerun()
