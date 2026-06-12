# app.py

import streamlit as st
from scraper.rag_pipeline import RAGPipeline

# ── Page config ────────────────────────────────
st.set_page_config(
    page_title="KU Faculty Chatbot",
    page_icon="🎓",
    layout="centered"
)

# ── Load pipeline ONCE per session ─────────────
# st.cache_resource ensures the model and ChromaDB
# connection are not reloaded on every message
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

# ── UI ─────────────────────────────────────────
st.title("🎓 KU Faculty Chatbot")
st.caption(
    "Ask questions about study programmes, exam rules, "
    "application procedures, and general faculty information."
)

# Initialise chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    if s["url"].startswith("http"):
                        st.markdown(f"- [{s['title']}]({s['url']}) — score: {s['score']:.3f}")
                    else:
                        st.markdown(f"- {s['title']} — score: {s['score']:.3f}")

# Chat input
question = st.chat_input("Ask a question about KU...")

if question:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = pipeline.ask(question)

        st.write(result["answer"])

        with st.expander("Sources", expanded=False):
            for s in result["sources"]:
                if s["url"].startswith("http"):
                    st.markdown(f"- [{s['title']}]({s['url']}) — score: {s['score']:.3f}")
                else:
                    st.markdown(f"- {s['title']} — score: {s['score']:.3f}")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
