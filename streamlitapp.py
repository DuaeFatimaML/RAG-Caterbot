import streamlit as st
import os

# Import your existing RAG functions from app.py
from ragapp2 import (
    rag_answer,
    BOT_NAME,
    BUSINESS_NAME,
    THEME_COLOR,
    WELCOME_MSG,
    VECTOR_STORE
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = f"{BOT_NAME} — {BUSINESS_NAME}",
    page_icon  = "🤖",
    layout     = "centered"
)

# ── Custom CSS to make it look like a chatbot ────────────────
st.markdown(f"""
<style>
  .stApp {{ background-color: #070b14; }}
  .chat-header {{
    background: linear-gradient(135deg, {THEME_COLOR}, #7c3aed);
    padding: 16px 20px;
    border-radius: 12px;
    color: white;
    font-family: sans-serif;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .user-msg {{
    background: linear-gradient(135deg, {THEME_COLOR}, #7c3aed);
    color: white;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    margin: 6px 0;
    max-width: 75%;
    margin-left: auto;
    font-size: 14px;
    line-height: 1.5;
  }}
  .bot-msg {{
    background: #1e293b;
    color: #e2e8f0;
    padding: 12px 16px;
    border-radius: 16px 16px 16px 4px;
    margin: 6px 0;
    max-width: 75%;
    border: 1px solid #334155;
    font-size: 14px;
    line-height: 1.5;
  }}
  .source-tag {{
    background: #1e3a5f;
    color: #93c5fd;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 99px;
    display: inline-block;
    margin: 2px;
  }}
  .msg-label {{
    font-size: 11px;
    color: #64748b;
    margin-bottom: 2px;
  }}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown(f"""
<div class="chat-header">
  🤖 <div>
    <strong style="font-size:16px">{BOT_NAME}</strong><br/>
    <span style="font-size:12px;opacity:.8">
      🟢 Online — {BUSINESS_NAME}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Check knowledge base ─────────────────────────────────────
if not os.path.exists(VECTOR_STORE):
    st.error("⚠️ Knowledge base not found. Run `python ingest.py` first.")
    st.stop()

# ── Session state: store chat history ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history  = []
    # Add welcome message
    st.session_state.messages.append({
        "role":    "bot",
        "content": WELCOME_MSG,
        "sources": []
    })

# ── Display all messages ─────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-label" style="text-align:right">You</div>
        <div class="user-msg">{msg["content"]}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-label">{BOT_NAME}</div>
        <div class="bot-msg">{msg["content"]}</div>
        """, unsafe_allow_html=True)

        # Show source tags
        if msg.get("sources"):
            src_html = "".join(
                f'<span class="source-tag">📄 {s}</span>'
                for s in msg["sources"]
            )
            st.markdown(src_html, unsafe_allow_html=True)

# ── Input box ─────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role":    "user",
        "content": user_input,
        "sources": []
    })
    st.session_state.history.append({
        "role":    "user",
        "content": user_input
    })

    # Get RAG answer
    with st.spinner("Searching documents..."):
        answer, sources = rag_answer(
            user_input,
            st.session_state.history[-10:]
        )

    # Add bot reply
    st.session_state.messages.append({
        "role":    "bot",
        "content": answer,
        "sources": sources
    })
    st.session_state.history.append({
        "role":    "assistant",
        "content": answer
    })

    st.rerun()