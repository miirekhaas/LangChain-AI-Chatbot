import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
    Create FAISS vector store from PDF pages using OpenRouter-compatible embeddings.
    """
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

        # OpenRouter API key setup
        openrouter_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            st.error("❌ OPENROUTER_API_KEY not found. Set it in Streamlit Secrets or environment variable.")
            return None

        os.environ["OPENAI_API_KEY"] = openrouter_key  # REQUIRED by langchain_openai
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

        st.write("✅ Step 3: OpenRouter API Key and Base URL set.")

        # ✅ Updated embedding model compatible with OpenRouter
        embeddings = OpenAIEmbeddings(
            model="thenlper/gte-large",  # or "nomic-ai/nomic-embed-text"
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_key
        )

        st.write("✅ Step 4: Embeddings initialized with OpenRouter model.")

        vector_store = FAISS.from_documents(docs, embeddings)
        st.write("✅ Step 5: Vector store created.")

        return vector_store

    except Exception as e:
        st.error(f"❌ Failed to create vector store: {e}")
        return None


def get_conversational_chain(vector_store):
    """
    Create a QA chatbot using OpenRouter-compatible LLM and a retriever.
    """
    try:
        if not vector_store:
            st.error("❌ Vector store is missing. Cannot create chatbot.")
            return None

        llm = ChatOpenAI(
            temperature=0.3,
            model="mistralai/mixtral-8x7b",  # You can also try "mistralai/mistral-7b-instruct"
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
