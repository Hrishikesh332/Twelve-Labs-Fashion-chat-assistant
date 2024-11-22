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
        
        # Print collection schema for debugging
        st.write("Collection Schema:", collection.schema)
        st.write("Collection Description:", collection.description)
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        # Search in Milvus collection
        results = collection.search(
            data=[question_embedding],  # Search vector
            anns_field="vector",       # Field to search
            param=search_params,       # Search parameters
            limit=3,                   # Top k
            output_fields=["*"]     # Get all available fields
        )

        # Debug: Print first hit structure
        if results and len(results) > 0 and len(results[0]) > 0:
            st.write("First hit structure:", dir(results[0][0]))
            st.write("First hit entity:", dir(results[0][0].entity))

        # Extract retrieved documents
        retrieved_documents = []
        for hit in results[0]:
            # Get all available fields from the entity
            entity_dict = {}
            for field_name in dir(hit.entity):
                if not field_name.startswith('_'):  # Skip private attributes
                    try:
                        value = getattr(hit.entity, field_name)
                        entity_dict[field_name] = value
                    except Exception as e:
                        continue
            
            st.write("Entity fields:", entity_dict)
            
            # Try to get the text content from available fields
            text_content = None
            if hasattr(hit.entity, 'text'):
                text_content = hit.entity.text
            elif hasattr(hit.entity, 'content'):
                text_content = hit.entity.content
            elif hasattr(hit.entity, 'data'):
                text_content = hit.entity.data
                
            if text_content:
                retrieved_documents.append((text_content, hit.distance))

        # If no results found
        if not retrieved_documents:
            return "I couldn't find any relevant information to answer your question. Could you please rephrase it?"

        # Convert retrieved documents to context string
        context = "\n".join([doc[0] for doc in retrieved_documents])

        # Define prompts
        SYSTEM_PROMPT = """
        You are an AI assistant. You will use the provided context to answer questions about Milvus.
        If the context doesn't contain relevant information, say so clearly.
        """

        USER_PROMPT = f"""
        Use the following context to answer the question:
        
        Context:
        {context}
        
        Question:
        {question}
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
        st.error(f"Error details: {str(e)}")
        return "I'm having trouble accessing the knowledge base. Could you share what fields are available in your Milvus collection?"

# Display collection information at startup
try:
    st.sidebar.write("Collection Information:")
    st.sidebar.write("Schema:", collection.schema)
    st.sidebar.write("Description:", collection.description)
except Exception as e:
    st.sidebar.write("Could not fetch collection info:", str(e))

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
