import streamlit as st
import os
from openai import RateLimitError
from dotenv import load_dotenv

from config.header import setup_chat_page_header
from config.rag_config import app_config

load_dotenv()

api_key = os.getenv("API_KEY")
ai_model = os.getenv("AI_MODEL")
    

client, embedder, collection = app_config(api_key=api_key)

setup_chat_page_header()

MAX_HISTORY = 6

def condense_question(question):
    history = st.session_state.messages[-MAX_HISTORY:]
    
    if not history:
        return question
    
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    response = client.chat.completions.create(
        model=ai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "Rewrite the latest question as a standalone question, using the "
                "conversation to resolve words like 'it', 'that' or 'them'. "
                "If it is already standalone or changes the topic, return it unchanged. "
                "Do not answer it. Return only the question."
            )},
            {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest question: {question}"},
        ],
    )
    return (response.choices[0].message.content or "").strip() or question
    
    
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
    

if question := st.chat_input("Ask about company policy: "):
    st.chat_message("user").write(question)
    
    history = st.session_state.messages[-MAX_HISTORY:]
    
    condensed_question = condense_question(question)
    
    question_embedding = embedder.encode([condensed_question])
    
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=4
    )
    
    retrieved_chunks = results["documents"][0]
    
    context = "\n\n---\n\n".join(retrieved_chunks)
        
    system_prompt = """
        Answer the question using only the context provided.
        If the context does not contain the answer, say you don't know.
    """
    response_guideline = """
        Make your response look a natural human response.
        Avoid these or any robot or chatbot like response:
        - Based on the context...
        - Based on the PDF document...
    """
    user_prompt = f"Context:\n{context}\n\nQuestion: {condensed_question}"
        
    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                    model=ai_model,
                    messages=[
                        {"role": "system", "content": system_prompt + response_guideline},
                        *history,
                        {"role": "user", "content": user_prompt},
                    ],
                )
        except RateLimitError:
            st.error("We are experiencing an issue. Please try again later.")
            st.stop()
    answer = response.choices[0].message.content
    st.chat_message("assistant").write(answer)
    
    # Append two messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
        
    