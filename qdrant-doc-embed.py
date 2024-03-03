import streamlit as st
from qdrant_client import QdrantClient
import openai
from dotenv import load_dotenv
import os

# Carga las variables de entorno
load_dotenv()

# Configura tus claves API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# Inicializa el cliente de OpenAI
openai.api_key = OPENAI_API_KEY

# Función para conectarse a Qdrant
def connect_to_qdrant():
    return QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY
    )

# Función para generar embeddings utilizando OpenAI
def generate_query_embedding(query):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[query]  # Nota que 'input' ahora debe ser una lista
    )
    # Accede directamente al atributo deseado de la respuesta
    embedding = response.data[0].embedding
    return embedding



# Función para buscar en Qdrant los CVs más relevantes basados en la pregunta
def search_in_qdrant(qdrant_client, query_embedding):
    search_result = qdrant_client.search(
        collection_name="DOCUMENT-EMBEDDINGS",
        query_vector=query_embedding,
        limit=5  # Ajusta este valor según necesites
    )
    return search_result

# Aplicación principal
def main():
    st.title('Buscador de CVs')

    # Conectarse a Qdrant
    qdrant_client = connect_to_qdrant()

    # Campo de entrada para la pregunta del usuario
    user_query = st.text_input("Introduce tu pregunta:")

    if user_query:
        # Generar embedding para la pregunta
        query_embedding = generate_query_embedding(user_query)

        # Realizar búsqueda en Qdrant
        results = search_in_qdrant(qdrant_client, query_embedding)

        if results:
            st.write("Resultados:")
            for idx, result in enumerate(results, start=1):
                st.write(f"{idx}. ID del documento: {result.id}, Puntuación: {result.score}")
        else:
            st.write("No se encontraron resultados relevantes.")

if __name__ == '__main__':
    main()
