# GUI and enviroment
from dotenv import load_dotenv
import streamlit as st
import os
from backend import get_pdf_text, get_vector_store

# embeddings and llms
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

# vector database
from langchain.vectorstores import Qdrant
import qdrant_client

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_vector_store(collection_name):
    # inicializar un cliente de Qdrant para interactuar con la base de datos vectorial
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    # utilizaremos este mecanismo de embedding
    embeddings = OpenAIEmbeddings()
    # creamos un objeto Qdrant, el cliente ya lo hemos definido y el método de embeddings también
    vector_store = Qdrant(
        client=client, 
        collection_name=collection_name,  # Usa el argumento para especificar el nombre de la colección
        embeddings=embeddings,
    )
    
    return vector_store


def main():
    load_dotenv()
    
    st.set_page_config(page_title="Ask Qdrant")
    st.header("Ask your remote database 💬")

    # Opción para seleccionar la colección de Qdrant
    available_collections = ['CVI-1', 'DOCUMENT-EMBEDDINGS'] 
    selected_collection = st.selectbox('Choose a Qdrant collection:', available_collections)

    # Crear vector store con la colección seleccionada
    vector_store = get_vector_store(selected_collection)
    
    # Crear cadena de recuperación y QA
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model='gpt-3.5-turbo-instruct'),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Mostrar entrada de usuario
    user_question = st.text_input("Ask a question:")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")

if __name__ == '__main__':
    main()