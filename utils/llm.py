import streamlit as st
import pypdf
import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

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
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Set up retriever
def get_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Return top 3 most relevant chunks
    )
    return retriever

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
def rag_pipeline(query, retriever):
    # Retrieve relevant chunks
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Prepare context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
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