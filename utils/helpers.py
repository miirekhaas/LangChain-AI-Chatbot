import os
import streamlit as st
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

# Set Google API Key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def load_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return pages

def create_vector_store(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    return chain
