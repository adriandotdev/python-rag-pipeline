from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st
import chromadb

def chunk_text(text, chunk_size=1000, overlap=200):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap

    return chunks

@st.cache_resource(show_spinner="Loading the policy assistant...")
def app_config(api_key: str):
    client = OpenAI(
        # base_url='https://openrouter.ai/api/v1', 
        # api_key=api_key
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required by the library, ignored by Ollama
    )
    
    # Extract text from the document
    reader = PdfReader("./employee_policy.pdf")
    document_text = ""

    for page in reader.pages:
        document_text += page.extract_text() + "\n"
    
    # Embedder and chroma client instances 
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("employee_policy")
    
    if collection.count() == 0:
        chunks = chunk_text(document_text, 800, 150)
        
        # print(f"{len(chunks)} chunks")
        # for i, chunk in enumerate(chunks[:3]):
        #     print(f"\n--- Chunk {i} ---\n{chunk}")

        embeddings = embedder.encode(chunks)
        
        collection.upsert(
            ids=[f"chunk-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist()
        )

    return client, embedder, collection