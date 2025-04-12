import streamlit as st
import pypdf
import os
import faiss
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# Define the path relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "data", "handbook.pdf")


# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split text into chunks
def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


#Embedding with HuggingFace model
@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

def embed_text(texts, tokenizer, model):
    import torch
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1).numpy()
    return normalize(embeddings)



# Create vector store
def create_vector_store(chunks):
    tokenizer, model = load_embedding_model()
    embeddings = embed_text(chunks, tokenizer, model)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return {"index": index, "texts": chunks, "model": model, "tokenizer": tokenizer}

# Set up retriever
def get_top_k(query, vector_store, k=3):
    tokenizer = vector_store["tokenizer"]
    model = vector_store["model"]
    q_embed = embed_text([query], tokenizer, model)
    D, I = vector_store["index"].search(q_embed, k)
    results = [vector_store["texts"][i] for i in I[0]]
    return results

# Query Mistral API
def query_mistral(prompt):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": "Bearer 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"}
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 500}
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.text}"

# RAG pipeline
def rag_pipeline(query, vector_store):

    # Retrieve relevant chunks
    chunks = get_top_k(query, vector_store)
    
    # Prepare context
    context = "\n\n".join(chunks)
    
    # Create a prompt that includes context
    prompt = f"""<s>[INST] You are an Academic City University assistant tasked with answering questions about the student handbook. 
    -You adapt your response based on the user's question.
    - If the question asks for a summary, respond briefly.
    - If the question asks for details or an explanation, elaborate clearly.

    
    Context:
    {context}
    
    Question: {query}
    
    Provide a comprehensive answer based only on the information in the context. If the information is not in the context, say "I don't have information about that in the student handbook." [/INST]</s>
    """
    
    # Generate response using Mistral
    response = query_mistral(prompt)
    
    return response