import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os
import streamlit as st
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings

# Load API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def get_gemini_embedding():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain.memory import ConversationBufferMemory

def load_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def create_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

def get_conversational_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
