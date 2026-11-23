"""
app.py
------
Streamlit frontend for the Smart FAQ Chatbot.
Connects the NLP backend (chatbot.py + preprocess.py) to a modern chat UI.
"""

import time
import streamlit as st
from chatbot import FAQChatbot
from preprocess import get_preprocessing_steps

# ── Page configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="Smart FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS – dark-blue / purple / white theme ─────────────────────────────
st.markdown(
    """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #12183a 50%, #1a0a2e 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #6d28d9 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(99,102,241,0.3);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.hero-logo {
    font-size: 3.5rem;
    line-height: 1;
}

.hero-title {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}

.hero-subtitle {
    color: #c7d2fe;
    font-size: 1rem;
    margin: 0.3rem 0 0 0;
}

.hero-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.6rem;
    flex-wrap: wrap;
}

.badge {
    background: rgba(255,255,255,0.15);
    color: #e0e7ff;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.2);
}

/* ── Chat container ── */
.chat-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 1.5rem;
    min-height: 420px;
    max-height: 540px;
    overflow-y: auto;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Scrollbar */
.chat-container::-webkit-scrollbar { width: 6px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }

/* ── Message bubbles ── */
.msg-row-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
    animation: slideInRight 0.3s ease;
}

.msg-row-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 1rem;
    animation: slideInLeft 0.3s ease;
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}

.bubble-user {
    background: linear-gradient(135deg, #3730a3, #6d28d9);
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 70%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(109,40,217,0.35);
}

.bubble-bot {
    background: rgba(30,58,138,0.6);
    color: #e0e7ff;
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1.1rem;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.6;
    border: 1px solid rgba(99,102,241,0.3);
    box-shadow: 0 4px 15px rgba(30,58,138,0.3);
}

.bubble-bot .meta-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.6rem;
    flex-wrap: wrap;
}

.meta-chip {
    font-size: 0.68rem;
    padding: 0.15rem 0.55rem;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.chip-category { background: rgba(99,102,241,0.3); color: #c7d2fe; }
.chip-high     { background: rgba(16,185,129,0.25); color: #6ee7b7; }
.chip-medium   { background: rgba(245,158,11,0.25); color: #fcd34d; }
.chip-low      { background: rgba(239,68,68,0.2);   color: #fca5a5; }
.chip-none     { background: rgba(107,114,128,0.3); color: #d1d5db; }

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 2px;
}

.avatar-bot  { background: linear-gradient(135deg,#1e3a8a,#3730a3); border: 2px solid rgba(99,102,241,0.5); }
.avatar-user { background: linear-gradient(135deg,#4f46e5,#7c3aed); border: 2px solid rgba(167,139,250,0.5); }

/* ── Welcome card ── */
.welcome-card {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #a5b4fc;
}

.welcome-card .wc-icon { font-size: 4rem; margin-bottom: 1rem; }
.welcome-card h3 { color: #e0e7ff; font-size: 1.3rem; margin-bottom: 0.5rem; }
.welcome-card p  { font-size: 0.9rem; color: #818cf8; }

/* ── Typing indicator ── */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #818cf8;
    font-size: 0.85rem;
    padding: 0.5rem 0;
}

.typing-dots span {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #6d28d9;
    animation: bounce 1.2s infinite;
}

.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%,80%,100% { transform: translateY(0); }
    40%          { transform: translateY(-8px); }
}

/* ── Input area ── */
.input-wrapper {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1rem 1.2rem;
}

/* ── Suggested questions ── */
.suggest-title {
    color: #818cf8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    margin-top: 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1340 0%, #12183a 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}

.sidebar-section {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.sidebar-section h4 {
    color: #c7d2fe;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.sidebar-section p, .sidebar-section li {
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.6;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(99,102,241,0.1);
    color: #94a3b8;
    font-size: 0.82rem;
}

.stat-val {
    color: #a5b4fc;
    font-weight: 600;
}

.cat-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #c7d2fe;
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.72rem;
    margin: 0.2rem 0.1rem;
}

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.92rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #6d28d9 !important;
    box-shadow: 0 0 0 2px rgba(109,40,217,0.25) !important;
}

.stTextInput > div > div > input::placeholder { color: #64748b !important; }

/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #3730a3, #6d28d9) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
    width: 100%;
}

.stButton > button:hover { opacity: 0.85 !important; }

/* Secondary / small buttons for suggested questions */
.stButton.suggest > button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #c7d2fe !important;
    font-size: 0.78rem !important;
    border-radius: 20px !important;
    padding: 0.3rem 0.8rem !important;
    font-weight: 400 !important;
}

/* Expander */
.streamlit-expanderHeader {
    color: #c7d2fe !important;
    background: rgba(99,102,241,0.08) !important;
    border-radius: 8px !important;
}

/* Success / info boxes */
.stAlert { border-radius: 10px !important; }

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 0.5rem;
    border: 1px solid rgba(99,102,241,0.15);
}

div[data-testid="stMetric"] label { color: #818cf8 !important; }
div[data-testid="stMetric"] div   { color: #e0e7ff !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # List of {"role", "content", "meta"}
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""       # Suggested question clicked by user
if "input_key" not in st.session_state:
    st.session_state.input_key = 0               # Reset input box after send


# ── Load chatbot (cached across reruns) ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chatbot():
    """Load and cache the FAQ chatbot engine."""
    return FAQChatbot()


def init_chatbot():
    """Initialize chatbot in session state with friendly error handling."""
    if st.session_state.chatbot is None:
        with st.spinner("🔧 Loading NLP engine & FAQ database…"):
            try:
                st.session_state.chatbot = load_chatbot()
            except Exception as exc:
                st.error(f"❌ Failed to load chatbot: {exc}")
                st.stop()


# ── Helper: render confidence chip ────────────────────────────────────────────
def confidence_chip_class(confidence: str) -> str:
    mapping = {"High": "chip-high", "Medium": "chip-medium", "Low": "chip-low"}
    return mapping.get(confidence, "chip-none")


# ── Helper: render a single chat message ──────────────────────────────────────
def render_message(role: str, content: str, meta: dict = None):
    """Render a user or bot chat bubble."""
    if role == "user":
        st.markdown(
            f"""
<div class="msg-row-user">
  <div class="bubble-user">{content}</div>
  <div class="avatar avatar-user" style="margin-left:0.6rem;">👤</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        chip_class = confidence_chip_class(meta.get("confidence", "")) if meta else "chip-none"
        meta_html = ""
        if meta and meta.get("found"):
            meta_html = f"""
<div class="meta-row">
  <span class="meta-chip chip-category">📂 {meta.get('category','')}</span>
  <span class="meta-chip {chip_class}">🎯 {meta.get('confidence','')} Confidence</span>
  <span class="meta-chip chip-none">📊 Score: {meta.get('score', 0):.0%}</span>
</div>"""
        st.markdown(
            f"""
<div class="msg-row-bot">
  <div class="avatar avatar-bot" style="margin-right:0.6rem;">🤖</div>
  <div class="bubble-bot">{content}{meta_html}</div>
</div>""",
            unsafe_allow_html=True,
        )


# ── Main app layout ────────────────────────────────────────────────────────────
def main():
    init_chatbot()
    bot: FAQChatbot = st.session_state.chatbot

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#c7d2fe;text-align:center;'>🤖 Smart FAQ Bot</h2>",
            unsafe_allow_html=True,
        )

        # About
        st.markdown(
            """
<div class='sidebar-section'>
  <h4>ℹ️ About</h4>
  <p>An AI-powered FAQ assistant using <strong style='color:#a5b4fc;'>TF-IDF</strong> and
  <strong style='color:#a5b4fc;'>Cosine Similarity</strong> to find the best answer to
  your college-related questions.</p>
</div>""",
            unsafe_allow_html=True,
        )

        # Stats
        total = bot.get_total_faqs()
        categories = bot.get_categories()
        st.markdown(
            f"""
<div class='sidebar-section'>
  <h4>📊 Statistics</h4>
  <div class='stat-row'><span>Total FAQs</span><span class='stat-val'>{total}</span></div>
  <div class='stat-row'><span>Categories</span><span class='stat-val'>{len(categories)}</span></div>
  <div class='stat-row'><span>Algorithm</span><span class='stat-val'>TF-IDF + CosSim</span></div>
  <div class='stat-row'><span>NLP Library</span><span class='stat-val'>NLTK</span></div>
</div>""",
            unsafe_allow_html=True,
        )

        # Categories
        cat_pills = "".join(f"<span class='cat-pill'>{c}</span>" for c in categories)
        st.markdown(
            f"""
<div class='sidebar-section'>
  <h4>🗂️ FAQ Categories</h4>
  {cat_pills}
</div>""",
            unsafe_allow_html=True,
        )

        # How it works
        st.markdown(
            """
<div class='sidebar-section'>
  <h4>⚙️ How It Works</h4>
  <ol style='color:#94a3b8;font-size:0.8rem;padding-left:1.1rem;'>
    <li>Your question is <strong style='color:#c7d2fe;'>cleaned</strong> using NLP</li>
    <li>Converted to a <strong style='color:#c7d2fe;'>TF-IDF vector</strong></li>
    <li><strong style='color:#c7d2fe;'>Cosine Similarity</strong> finds the best match</li>
    <li>Answer returned with <strong style='color:#c7d2fe;'>confidence score</strong></li>
  </ol>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Clear chat button in sidebar
        if st.button("🗑️ Clear Chat History", key="sidebar_clear"):
            st.session_state.chat_history = []
            st.session_state.input_key += 1
            st.rerun()

        # Chat count
        msg_count = len(st.session_state.chat_history) // 2
        st.markdown(
            f"<p style='text-align:center;color:#64748b;font-size:0.75rem;'>"
            f"💬 {msg_count} conversation(s) this session</p>",
            unsafe_allow_html=True,
        )

    # ── MAIN COLUMN ────────────────────────────────────────────────────────────
    # Header banner
    st.markdown(
        """
<div class="hero-banner">
  <div class="hero-logo">🤖</div>
  <div>
    <p class="hero-title">Smart FAQ Chatbot</p>
    <p class="hero-subtitle">AI-Powered College Question Answering System</p>
    <div class="hero-badges">
      <span class="badge">🧠 TF-IDF + Cosine Similarity</span>
      <span class="badge">🔤 NLTK NLP Pipeline</span>
      <span class="badge">⚡ Real-time Matching</span>
      <span class="badge">📚 College FAQs</span>
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Chat display area ──────────────────────────────────────────────────────
    chat_placeholder = st.empty()

    with chat_placeholder.container():
        st.markdown('<div class="chat-container" id="chat-box">', unsafe_allow_html=True)

        if not st.session_state.chat_history:
            # Welcome message
            st.markdown(
                """
<div class='welcome-card'>
  <div class='wc-icon'>👋</div>
  <h3>Welcome! I'm your Smart FAQ Assistant</h3>
  <p>Ask me anything about college admissions, courses, fees, examinations,
     placements, scholarships, campus facilities, and more!</p>
  <p style='margin-top:0.5rem;'>Type your question below or click a suggested question to get started.</p>
</div>""",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_history:
                render_message(msg["role"], msg["content"], msg.get("meta"))

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Input area ────────────────────────────────────────────────────────────
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    col_input, col_send = st.columns([5, 1])

    with col_input:
        # Pre-fill with suggested question if one was clicked
        default_val = st.session_state.pending_question
        user_input = st.text_input(
            label="Ask a question",
            value=default_val,
            placeholder="e.g. What is the admission process for engineering?",
            label_visibility="collapsed",
            key=f"user_input_{st.session_state.input_key}",
        )

    with col_send:
        send_clicked = st.button("➤ Send", key="send_btn")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Suggested questions ───────────────────────────────────────────────────
    samples = bot.get_sample_questions(n=8)
    if samples:
        st.markdown('<p class="suggest-title">💡 Suggested Questions</p>', unsafe_allow_html=True)
        # Render in rows of 4
        for row_start in range(0, len(samples), 4):
            row_items = samples[row_start : row_start + 4]
            cols = st.columns(len(row_items))
            for col, question in zip(cols, row_items):
                with col:
                    # Truncate for button label
                    label = question if len(question) <= 55 else question[:52] + "…"
                    if st.button(label, key=f"suggest_{row_start}_{question[:20]}"):
                        st.session_state.pending_question = question
                        st.rerun()

    # ── Clear chat (main area) ────────────────────────────────────────────────
    if st.session_state.chat_history:
        _, col_clear = st.columns([5, 1])
        with col_clear:
            if st.button("🗑️ Clear", key="main_clear"):
                st.session_state.chat_history = []
                st.session_state.input_key += 1
                st.rerun()

    # ── NLP Preprocessing expander ────────────────────────────────────────────
    if st.session_state.chat_history:
        last_user = next(
            (m for m in reversed(st.session_state.chat_history) if m["role"] == "user"),
            None,
        )
        if last_user:
            with st.expander("🔬 View NLP Preprocessing Steps (last query)", expanded=False):
                steps = get_preprocessing_steps(last_user["content"])
                cols = st.columns(3)
                labels = [
                    ("Original Input", "original"),
                    ("Lowercase", "lowercase"),
                    ("Cleaned Text", "cleaned"),
                ]
                for col, (label, key) in zip(cols, labels):
                    with col:
                        st.markdown(f"**{label}**")
                        st.code(steps.get(key, ""), language=None)

                cols2 = st.columns(3)
                labels2 = [
                    ("Tokenized", "tokenized"),
                    ("Stopwords Removed", "stopwords_removed"),
                    ("Lemmatized (Final)", "lemmatized"),
                ]
                for col, (label, key) in zip(cols2, labels2):
                    with col:
                        st.markdown(f"**{label}**")
                        val = steps.get(key, [])
                        if isinstance(val, list):
                            st.code(str(val), language=None)
                        else:
                            st.code(val, language=None)

    # ── Process user input ────────────────────────────────────────────────────
    # Trigger when Send button clicked OR Enter pressed
    should_process = send_clicked and user_input.strip()

    if should_process:
        query = user_input.strip()

        # Clear pending suggested question
        st.session_state.pending_question = ""

        # Add user message to history
        st.session_state.chat_history.append(
            {"role": "user", "content": query, "meta": {}}
        )

        # Show typing indicator briefly
        typing_slot = st.empty()
        typing_slot.markdown(
            """
<div class='typing-indicator'>
  <span>🤖 Thinking</span>
  <div class='typing-dots'>
    <span></span><span></span><span></span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )
        time.sleep(0.8)  # Simulate processing delay for UX
        typing_slot.empty()

        # Get response from NLP engine
        try:
            result = bot.get_response(query)
        except Exception as exc:
            result = {
                "answer": f"⚠️ An error occurred: {exc}",
                "matched_question": "",
                "category": "",
                "score": 0.0,
                "confidence": "No Match",
                "found": False,
            }

        # Build bot response message
        answer_html = result["answer"]
        if result.get("found") and result.get("matched_question"):
            answer_html = (
                f"{result['answer']}<br><br>"
                f"<em style='color:#6366f1;font-size:0.78rem;'>"
                f"📌 Matched: \"{result['matched_question']}\"</em>"
            )

        st.session_state.chat_history.append(
            {
                "role": "bot",
                "content": answer_html,
                "meta": {
                    "category": result.get("category", ""),
                    "confidence": result.get("confidence", ""),
                    "score": result.get("score", 0.0),
                    "found": result.get("found", False),
                },
            }
        )

        # Advance input key to clear the text box
        st.session_state.input_key += 1
        st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

