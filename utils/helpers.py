import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain


def load_pdf_from_upload(uploaded_file):
    """
    Handles a Streamlit uploaded_file object, saves to a temp file, and loads pages using PyPDFLoader.
    """
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
    """
    Loads a local PDF file and returns a list of LangChain Document objects.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return pages
    except Exception as e:
        st.error(f"❌ Failed to load PDF from path: {e}")
        return []


def create_vector_store(pages):
    """
    Creates a FAISS vector store from a list of Document objects using Gemini embeddings.
    """
    try:
        if not pages:
            st.error("❌ No pages found to process.")
            return None

        # 1. Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)

        if not docs:
            st.error("❌ Failed to split documents into chunks.")
            return None

        # 2. Load Gemini API key
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ GOOGLE_API_KEY not found. Set it in Streamlit Secrets or as an environment variable.")
            return None
        os.environ["GOOGLE_API_KEY"] = api_key

        # 3. Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store

    except Exception as e:
        st.error(f"❌ Failed to create vector store: {e}")
        return None


def get_conversational_chain(vector_store):
    """
    Creates a conversational QA chain using Gemini and the provided vector store.
    """
    try:
        if vector_store is None:
            st.error("❌ Vector store is missing. Cannot create chat chain.")
            return None

        llm = ChatGoogleGenerativeAI(model="models/chat-bison-001", temperature=0.3)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        return chain

    except Exception as e:
        st.error(f"❌ Failed to initialize conversational chain: {e}")
        return None
