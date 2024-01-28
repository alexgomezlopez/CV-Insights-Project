# GUI and enviroment
from dotenv import load_dotenv
import streamlit as st
import os

# eat pdfs
from PyPDF2 import PdfReader

# embeddings and llms
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import openai

# vector database
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import qdrant_client
import json

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store():
    client = QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings()
    
    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store



def main():
    load_dotenv()
    st.set_page_config(page_title="Upload PDF to Qdrant database", page_icon=":books:")
    st.header('Upload your PDF documents')

    # Widget para subir PDFs
    uploaded_files = st.file_uploader("Sube tus documentos PDF aquí", accept_multiple_files=True, type=["pdf"])
    
    if uploaded_files:
        # Inicializar Qdrant Vector Store
        vector_store = get_vector_store()
        
        for uploaded_file in uploaded_files:
            # Convertir PDF a texto
            raw_text = get_pdf_text([uploaded_file])
            
            # Obtener chunks del texto
            text_chunks = get_text_chunks(raw_text)
            
            # Procesar y subir los chunks al vector store
            for chunk in text_chunks:
                # Obtener el embedding del chunk de texto
                embedding = vector_store.embeddings(chunk)
                
                # Subir el embedding al vector store
                vector_store.upsert(point_id=None,  # Puedes usar None para auto-generar un ID, o proporcionar un ID específico
                                    embedding_vector=embedding,
                                    payload=None)  # Puedes agregar metadatos si lo deseas
            
        st.success("Documentos procesados y vectores subidos correctamente.")

if __name__ == '__main__':
    main()