import streamlit as st
import time
from twelvelabs import TwelveLabs
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from urllib.parse import urlparse
import uuid
from dotenv import load_dotenv
import os
from pymilvus import MilvusClient
from pymilvus import connections
from pymilvus import (
    FieldSchema, DataType, 
    CollectionSchema, Collection,
    utility
)
from openai import OpenAI
import json

# Load environment variables
load_dotenv()
TWELVELABS_API_KEY = os.getenv('TWELVELABS_API_KEY')
MILVUS_DB_NAME = os.getenv('MILVUS_DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
openai_client = OpenAI()

# Milvus Connection
connections.connect(
    uri=URL,
    token=TOKEN
)

# Check if collection exists
if utility.has_collection(COLLECTION_NAME):
    collection = Collection(COLLECTION_NAME)
    print(f"Using existing collection: {COLLECTION_NAME}")
else:
    # Define fields for schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]
    
    # Create schema with dynamic fields for metadata
    schema = CollectionSchema(
        fields=fields,
        enable_dynamic_field=True
    )
    
    collection = Collection(COLLECTION_NAME, schema)
    print(f"Created new collection: {COLLECTION_NAME}")
    
    if not collection.has_index():
        collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        print("Created index for the new collection")

collection.load()

# Function to generate embeddings using ada-002 (1024 dimensions)
def emb_text(text):
    result = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",  # Using text-embedding-3-small
        dimensions=1024  # Explicitly setting dimensions to 1024
    )
    return result.data[0].embedding

def get_rag_response(question):
    try:
        # Generate embedding for the question
        question_embedding = emb_text(question)
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        # Search in Milvus collection
        search_results = collection.search(
            data=[question_embedding],  # Search vector
            anns_field="vector",       # Field to search
            param=search_params,       # Search parameters
            limit=3,                   # Top k
            output_fields=["text"]     # Fields to return
        )

        # Convert SequenceIterator to list to make it iterable
        results = list(search_results)
        
        # Extract retrieved documents
        retrieved_documents = []
        for hit in results[0]:
            try:
                text = str(hit.get('text', ''))  # Convert to string and provide default value
                distance = float(hit.get('distance', 0.0))  # Convert to float and provide default
                if text:  # Only add if text exists
                    retrieved_documents.append((text, distance))
            except AttributeError:
                # If hit is not a dictionary, try accessing as object
                text = getattr(hit.entity, 'text', '')
                distance = getattr(hit, 'distance', 0.0)
                if text:
                    retrieved_documents.append((text, distance))

        # Debug information
        st.write("Number of results found:", len(retrieved_documents))
        
        # If no results found
        if not retrieved_documents:
            return "I couldn't find any relevant information to answer your question. Please try asking something else."

        # Convert retrieved documents to context string
        context = "\n".join([doc[0] for doc in retrieved_documents])

        # Define prompts
        SYSTEM_PROMPT = """
        You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
        """

        USER_PROMPT = f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """

        # Generate response using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )

        return response.choices[0].message.content
    except Exception as e:
        import traceback
        st.error(f"An error occurred: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return f"I encountered an error while processing your question. Please try again."

# Streamlit UI
st.title("📚 RAG Chatbot with Milvus")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Milvus..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_rag_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with info
with st.sidebar:
    st.title("About")
    st.markdown("""
    This chatbot uses:
    - Milvus for vector storage and similarity search
    - OpenAI for embeddings and text generation
    - RAG (Retrieval-Augmented Generation) architecture
    
    Ask questions about Milvus to get informed responses based on the documentation!
    """)
