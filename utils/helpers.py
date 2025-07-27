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
    Save uploaded PDF to a temp file and load its pages.
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
    Load local PDF by path.
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
    Create FAISS vector store from PDF pages using Gemini embeddings.
    """
    try:
        if not pages:
            st.error("❌ No pages found to process.")
            return None

        # Step 1: Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        if not docs:
            st.error("❌ Failed to split documents into chunks.")
            return None

        # Step 2: Get API Key
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ GOOGLE_API_KEY not found. Set it in Streamlit Secrets or as an environment variable.")
            return None
        os.environ["GOOGLE_API_KEY"] = api_key

        # Step 3: Create embeddings + vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store

    except Exception as e:
        st.error(f"❌ Failed to create vector store: {e}")
        return None
        def create_vector_store(pages):
    try:
        if not pages:
            st.error("❌ No pages found to process.")
            return None

        st.write("✅ Step 1: Pages received:", len(pages))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)

        st.write("✅ Step 2: Docs after split:", len(docs))

        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ GOOGLE_API_KEY not found. Set it in Streamlit Secrets or as an environment variable.")
            return None
        os.environ["GOOGLE_API_KEY"] = api_key

        st.write("✅ Step 3: API Key is set.")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.write("✅ Step 4: Embeddings initialized.")

        vector_store = FAISS.from_documents(docs, embeddings)
        st.write("✅ Step 5: Vector store created.")

        return vector_store

    except Exception as e:
        st.error(f"❌ Failed to create vector store: {e}")
        return None



def get_conversational_chain(vector_store):
    """
    Create QA chain using Gemini chat model + vector store retriever.
    """
    try:
        if not vector_store:
            st.error("❌ Vector store is missing. Cannot create chatbot.")
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
