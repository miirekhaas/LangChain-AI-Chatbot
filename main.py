import streamlit as st
from utils.helpers import create_vector_store, get_conversational_chain, load_pdf_text
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="LangChain Chatbot", layout="wide")

st.title("🧠 AI PDF Chatbot")
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        raw_text = load_pdf_text(uploaded_file)
        vector_store = create_vector_store(raw_text)
        st.session_state.chain = get_conversational_chain(vector_store)
        st.success("✅ Document loaded and indexed!")

if "chain" in st.session_state:
    question = st.text_input("Ask a question about the document:")
    if question:
        response = st.session_state.chain.run(question)
        st.write("🤖", response)
