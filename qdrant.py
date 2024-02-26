# Se sube un documento, se genera el embedding con open ai y se hace una pregunta.

import os
from PyPDF2 import PdfReader
import re
import openai
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client import QdrantClient
#import qdrant_client
from qdrant_client.http import models
from dotenv import load_dotenv
import streamlit as st


def get_pdf_text(uploaded_file):
    """Extrae el texto de un archivo PDF cargado como UploadedFile en Streamlit."""
    text = ""
    try:
        # No es necesario abrir el archivo con 'open', PdfReader puede manejar el buffer directamente
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        text = re.sub(r'\s+', ' ', text).strip()  # Normaliza y limpia el texto
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
    return text

def get_text_chunks(text):
    chunks = []
    while len(text) > 500:
        last_period_index = text[:500].rfind('.')
        if last_period_index == -1:
            last_period_index = 500
        chunks.append(text[:last_period_index])
        text = text[last_period_index+1:]
    chunks.append(text)
    return chunks

def qdrant_connection():
    qdrant_client = QdrantClient(
        url = os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    qdrant_client.recreate_collection(
        collection_name="demo",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    return qdrant_client

def generate_embeddings(chunks):
    points = []
    i = 1
    for chunk in chunks:
        i += 1
        
        embeddings = openai.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        ).data[0].embedding

        points.append(PointStruct(id=i, vector=embeddings, payload={"text": chunk}))
    
    return points

def index_embeddings(points, qdrant_client):
    operation_info = qdrant_client.upsert(
        collection_name="demo",
        wait=True,
        points=points
    )

    return operation_info

def create_answer_with_context(query, qdrant_client):
    embeddings = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    #qdrant_client = qdrant_connection()  # Crear una instancia de QdrantClient

    search_result = qdrant_client.search(
        collection_name="demo",
        query_vector=embeddings, 
        limit=3
    )

    prompt = """You are a helpful HR assistant who answers 
                questions in brief based on the context below.
                Context:\n"""
    for result in search_result:
        prompt += result.payload['text'] + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"

    completion = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content


# def create_answer_with_context(query):
#     embeddings = openai.embeddings.create(
#         input=query,
#         model="text-embedding-3-small"
#     ).data[0].embedding

#     search_result = qdrant_client.search(
#         collection_name="demo",
#         query_vector=embeddings, 
#         limit=3
#     )

#     prompt = """You are a helpful HR assistant who answers 
#                 questions in brief based on the context below.
#                 Context:\n"""
#     for result in search_result:
#         prompt += result.payload['text'] + "\n---\n"
#     prompt += "Question:" + query + "\n---\n" + "Answer:"

#     completion = openai.chat.completions.create(
#         model="gpt-4-0125-preview",
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )

#     return completion.choices[0].message.content


# APLICACION PRINCIPAL
def main():
    load_dotenv()

    st.title('CV Insights with RAG & Qdrant')

    qdrant_client = qdrant_connection()

    uploaded_file = st.file_uploader("Carga un CV en formato PDF", type="pdf")
    if uploaded_file is not None:
        text = get_pdf_text(uploaded_file)
        #if st.button('Procesar Texto'):
        chunks = get_text_chunks(text)
        st.write(f"Se han extraído {len(chunks)} fragmentos del texto.")
        
        # Generar embeddings para cada fragmento de texto
        points = generate_embeddings(chunks)
        
        # Indexar los embeddings generados en Qdrant
        index_embeddings(points, qdrant_client)
        
        st.write("Los embeddings se han generado y indexado correctamente.")

        # Mostrar entrada de usuario
        user_question = st.text_input("Ask a question:")
        answer = create_answer_with_context(user_question, qdrant_client)  # Corregido aquí
        st.write(f"Answer: {answer}")

if __name__ == '__main__':
    main()

