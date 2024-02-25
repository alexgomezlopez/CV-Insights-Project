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

def load_context_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        context = file.read()
    return context

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
        llm=OpenAI(model='babbage-002'),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # # Cargar contexto desde el archivo
    # context_file_path = 'context.txt'
    # context = load_context_from_file(context_file_path)

    # Mostrar entrada de usuario
    user_question = st.text_input("Ask a question:")
    if user_question:
        st.write(f"Question: {user_question}")

        # # Concatenar el prompt con el contexto y la pregunta del usuario
        # input_text = f"Question from the user: {user_question}"

        # Concatenar el prompt con el contexto y la pregunta del usuario
        input_text = f"Question from the user: {user_question}"
        
        # Obtener respuesta del modelo de lenguaje con el prompt y la pregunta del usuario
        answer = qa.run(input_text)
        
        st.write(f"Answer: {answer}")

if __name__ == '__main__':
    main()
