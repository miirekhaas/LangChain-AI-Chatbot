import os
import streamlit as st
from utils.helpers import (
    load_pdf_from_upload,
    create_vector_store,
    get_conversational_chain
)

# App config
st.set_page_config(page_title="📄 OpenRouter PDF Chatbot", layout="wide")
st.title("📄 OpenRouter Chatbot with PDF")

# Set OpenRouter API Key
openrouter_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    st.error("❌ OPENROUTER_API_KEY not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = openrouter_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="pdf_upload")

if uploaded_file:
    st.success(f"✅ PDF uploaded: {uploaded_file.name}")

    with st.spinner("🔍 Processing your PDF..."):
        try:
            pages = load_pdf_from_upload(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error loading PDF: {e}")
            st.stop()

    if not pages:
        st.error("❌ No text extracted from PDF.")
        st.stop()

    st.info(f"📄 Loaded {len(pages)} page(s).")

    with st.spinner("🔎 Creating vector store..."):
        try:
            vector_store = create_vector_store(pages)
        except Exception as e:
            st.error(f"❌ Error creating vector store: {e}")
            st.stop()

    if not vector_store:
        st.error("❌ Vector store creation failed.")
        st.stop()

    with st.spinner("⚙️ Initializing chatbot..."):
        try:
            chain = get_conversational_chain(vector_store)
        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {e}")
            st.stop()

    if not chain:
        st.error("❌ Could not load chatbot chain.")
        st.stop()

    st.success("✅ Chatbot is ready! Ask your questions below.")
    chat_history = []

    query = st.text_input("💬 Ask something from your PDF:", key="user_query")

    if query:
        with st.spinner("✍️ Generating answer..."):
            try:
                result = chain({
                    "question": query,
                    "chat_history": chat_history
                })
                answer = result.get("answer", "No answer found.")
                chat_history.append((query, answer))
                st.markdown(f"**🧠 Answer:** {answer}")
            except Exception as e:
                st.error(f"❌ Error during response generation: {e}")
