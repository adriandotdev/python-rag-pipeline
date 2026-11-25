import os
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

api_key = os.getenv("API_KEY")
ai_model = os.getenv("AI_MODEL")

client = OpenAI(
    base_url='https://openrouter.ai/api/v1', 
    api_key=api_key
)

reader = PdfReader("./employee_policy.pdf")

document_text = ""

for page in reader.pages:
    document_text += page.extract_text() + "\n"
    

# Function that chunks the text
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


# Embed the chunks
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

    print(f"Stored {collection.count()} chunks")

question = None | str

while True:
    question = input("\n\nAsk about company policy: ")

    if question.lower() == "exit":
        break
    
    question_embedding = embedder.encode([question])

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=4
    )

    retrieved_chunks = results["documents"][0]

    context = "\n\n---\n\n".join(retrieved_chunks)

    system_prompt = (
        "Answer the question using only the context provided. "
        "If the context does not contain the answer, say you don't know."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=ai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    print(f"\nAnswer: {response.choices[0].message.content}")