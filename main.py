import streamlit as st
from utils.helpers import load_pdf_text, create_vector_store, get_conversational_chain

st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")

st.title("📄 Gemini Chatbot with PDF")

# File upload
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    with st.spinner("Processing PDF..."):
        pages = load_pdf_text(pdf)
        vector_store = create_vector_store(pages)
        chain = get_conversational_chain(vector_store)
        st.success("PDF processed successfully!")

    chat_history = []

    # User input
    query = st.text_input("Ask your PDF:")
    if query:
        with st.spinner("Generating answer..."):
            result = chain({"question": query, "chat_history": chat_history})
            response = result['answer']
            chat_history.append((query, response))
            st.markdown(f"**Answer:** {response}")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success(f"✅ PDF uploaded: {uploaded_file.name}")
    pages = load_pdf_text(uploaded_file)

    if pages:
        st.info(f"Loaded {len(pages)} page(s)")
    else:
        st.error("❌ Failed to load PDF.")
