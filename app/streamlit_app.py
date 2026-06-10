# import streamlit as st
# from chatbot import retrieve_context, ask_ollama

# st.set_page_config(
#     page_title="KU MIDS Chatbot",
#     page_icon="🎓",
#     layout="centered"
# )

# st.title("🎓 KU MIDS Chatbot")
# st.write(
#     "Ask questions about MIDS, Data Science programs, professors, applications, and student information."
# )

# question = st.chat_input("Ask a question:")

# if question:
#     with st.chat_message("user"):
#         st.write(question)

#     with st.chat_message("assistant"):
#         with st.spinner("Searching the KU/MIDS knowledge base..."):
#             context, sources = retrieve_context(question)
#             answer = ask_ollama(question, context)

#         st.write(answer)

#         with st.expander("Sources used"):
#             for source in sources:
#                 st.write(f"- {source}")

import streamlit as st

from chatbot import (
    retrieve_context,
    ask_llm,
    set_llm_mode
)

st.set_page_config(
    page_title="KU MIDS Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 KU MIDS Chatbot")

llm_choice = st.sidebar.radio(
    "Choose LLM",
    [
        "Hugging Face (Llama 3.1 8B)",
        "Local Ollama (gemma3:1b)"
    ]
)

use_hf = (
    llm_choice == "Hugging Face (Llama 3.1 8B)"
)

set_llm_mode(use_hf)

st.sidebar.markdown("---")

if use_hf:
    st.sidebar.success("Using Hugging Face Llama 3.1 8B")
else:
    st.sidebar.info("Using Local Ollama gemma3:1b")

st.write(
    "Ask questions about MIDS, Data Science programs, professors, applications, and student information."
)

question = st.chat_input("Ask a question:")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the KU/MIDS knowledge base..."):
            context, sources = retrieve_context(question)
            answer = ask_llm(question, context)

        st.write(answer)

        with st.expander("Sources used"):
            for source in sources:
                st.write(f"- {source}")