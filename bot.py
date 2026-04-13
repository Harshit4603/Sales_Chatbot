import streamlit as st
from app import process_query

# Page config + hide default Streamlit UI
st.set_page_config(page_title="🤖 Chatbot", layout="wide", initial_sidebar_state="collapsed")

# Hide all default headers, footers, toolbar, etc.
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Session state to control loading screen
if "show_loading" not in st.session_state:
    st.session_state.show_loading = True

# ========================== FULL SCREEN LOADING SCREEN ==========================
if st.session_state.show_loading:

    st.markdown("""
    <style>
        /* Full background image for the whole screen */
        .stApp {
            background-image: url("sample_img.jpg");   /* ←←← CHANGE THIS TO YOUR IMAGE NAME */
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Dark overlay */
        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100vh;
            background: rgba(0, 0, 0, 0.68);   /* Adjust 0.68 if too dark or too light */
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 999;
        }

        .loading-box {
            text-align: center;
            color: white;
            max-width: 800px;
        }

        @keyframes fadeInFromBottom {
            from { opacity: 0; transform: translateY(100px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .primary-text {
            font-size: 3.4rem;
            font-weight: 700;
            animation: fadeInFromBottom 2s ease forwards;
            margin-bottom: 25px;
            text-shadow: 0 5px 20px rgba(0,0,0,0.8);
        }

        .typewriter {
            font-size: 1.75rem;
            color: #00ffbb;
            overflow: hidden;
            white-space: nowrap;
            border-right: 5px solid #00ffbb;
            animation: typing 4.5s steps(50, end) forwards, blink-caret 0.75s step-end infinite;
            margin-bottom: 50px;
        }

        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }

        @keyframes blink-caret {
            50% { border-color: transparent; }
        }
    </style>
    """, unsafe_allow_html=True)

    # The actual loading content
    st.markdown("""
    <div class="overlay">
        <div class="loading-box">
            <h1 class="primary-text">Welcome to Your AI Chatbot</h1>
            <div class="typewriter">Initializing the smartest assistant...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Start button (placed in normal Streamlit flow so it stays clickable)
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)   # push button down
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if st.button("🚀 Start Chatting", type="primary", use_container_width=True):
            st.session_state.show_loading = False
            st.rerun()

# ========================== MAIN CHAT SCREEN (appears after clicking Start) ==========================
else:
    st.title("🤖 Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_query(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})