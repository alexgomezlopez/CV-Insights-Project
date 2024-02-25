# enviroment
from dotenv import load_dotenv
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