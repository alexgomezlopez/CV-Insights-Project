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

def get_vector_store():
    # inicializar un cliente de Qdrant para interactuar con la base de datos vectorial
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    #utilizaremos este mecanismo de embedding
    embeddings = OpenAIEmbeddings()
    # creamos un objeto Qdrant, el cliente ya lo hemos definido y el método de embeddings también
    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Ask Qdrant")
    st.header("Ask your remote database 💬")
    
    # create vector store
    vector_store = get_vector_store()
    
    # create chain 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model = 'gpt-3.5-turbo-instruct'),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # show user input
    user_question = st.text_input("Ask a question:")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")
    
        
if __name__ == '__main__':
    main()