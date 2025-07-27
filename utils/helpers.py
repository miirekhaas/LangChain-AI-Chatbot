import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


def load_pdf_from_upload(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        return pages

    except Exception as e:
        st.error(f"❌ Failed to load PDF from upload: {e}")
        return []


def load_pdf_from_path(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return pages
    except Exception as e:
        st.error(f"❌ Failed to load PDF from path: {e}")
        return []


def create_vector_store(pages):
    try:
        if not pages:
            st.error("❌ No pages found to process.")
            return None

        st.write("✅ Step 1: Pages received:", len(pages))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        if not docs:
            st.error("❌ Failed to split documents into chunks.")
            return None

        st.write("✅ Step 2: Docs after split:", len(docs))

        # ✅ No API key needed here
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.write("✅ Step 3: Local HuggingFace Embeddings initialized.")

        vector_store = FAISS.from_documents(docs, embeddings)
        st.write("✅ Step 4: Vector store created using FAISS.")

        return vector_store

    except Exception as e:
        st.error(f"❌ Failed to create vector store: {e}")
        return None


def get_conversational_chain(vector_store):
    try:
        if not vector_store:
            st.error("❌ Vector store is missing. Cannot create chatbot.")
            return None

        llm = ChatOpenAI(
    temperature=0.3,
    model="openrouter/openai/gpt-3.5-turbo",  # ✅ Use a valid model ID
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY"),
)



        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )

        return chain

    except Exception as e:
        st.error(f"❌ Failed to initialize conversational chain: {e}")
        return None
