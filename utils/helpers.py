import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain


import tempfile
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_text(uploaded_file):
    try:
        # Save uploaded PDF to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Load PDF pages using LangChain
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        return pages

    except Exception as e:
        return f"Error loading PDF: {e}"


def load_pdf_text(pdf_path):
    """
    Loads a PDF file and extracts its content using PyPDFLoader.
    Returns a list of Document objects.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return pages
    except Exception as e:
        st.error(f"❌ Failed to load PDF: {e}")
        return []


def create_vector_store(pages):
    """
    Splits the PDF pages into smaller chunks and creates a FAISS vector store using Gemini embeddings.
    Returns the vector store object.
    """
    try:
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)

        # Load Gemini API key
        api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
        if not api_key:
            st.error("❌ GOOGLE_API_KEY not found. Please set it in Streamlit Secrets or as an environment variable.")
            return None

        os.environ["GOOGLE_API_KEY"] = api_key

        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store

    except Exception as e:
        st.error(f"❌ Failed to create vector store: {e}")
        return None


def get_conversational_chain(vector_store):
    """
    Creates a conversational retrieval chain using Gemini LLM.
    Returns the chain object.
    """
    try:
        if vector_store is None:
            st.error("❌ Vector store is not available.")
            return None

        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        return chain

    except Exception as e:
        st.error(f"❌ Failed to initialize conversational chain: {e}")
        return None
