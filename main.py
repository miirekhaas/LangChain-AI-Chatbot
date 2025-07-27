import streamlit as st
from utils.helpers import (
    load_pdf_from_upload,
    create_vector_store,
    get_conversational_chain
)

# Streamlit app config
st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
st.title("📄 Gemini Chatbot with PDF")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success(f"✅ PDF uploaded: {uploaded_file.name}")

    with st.spinner("🔍 Processing PDF..."):
        pages = load_pdf_from_upload(uploaded_file)

        if pages:
            st.info(f"📄 Loaded {len(pages)} page(s).")
            vector_store = create_vector_store(pages)

            if vector_store:
                chain = get_conversational_chain(vector_store)

                if chain:
                    st.success("✅ Chatbot is ready! Start asking your questions.")
                    chat_history = []

                    query = st.text_input("💬 Ask something from your PDF:")

                    if query:
                        with st.spinner("✍️ Generating answer..."):
                            result = chain({
                                "question": query,
                                "chat_history": chat_history
                            })

                            answer = result['answer']
                            chat_history.append((query, answer))

                            st.markdown(f"**🧠 Answer:** {answer}")
                else:
                    st.error("❌ Failed to initialize chatbot.")
            else:
                st.error("❌ Failed to create vector store.")
        else:
            st.error("❌ Failed to extract content from PDF.")
