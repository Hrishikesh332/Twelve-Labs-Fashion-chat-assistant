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
from pymilvus import connections, Collection, utility
from openai import OpenAI


load_dotenv()
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TWELVELABS_API_KEY = os.getenv('TWELVELABS_API_KEY')


st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #fafafa;
    }
    
    /* Header styling */
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e1e1e;
        font-weight: 700;
        padding-bottom: 2rem;
    }
    
    /* Chat container styling */
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Product card styling */
    .product-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #FF4B6B;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FF3358;
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f1f1;
    }
    
    /* Video player styling */
    .stVideo {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Custom divider */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #FF4B6B 0%, #FF8E53 100%);
        margin: 1rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize connections
openai_client = OpenAI()
connections.connect(uri=URL, token=TOKEN)
collection = Collection(COLLECTION_NAME)
collection.load()

# Embedding function
def emb_text(text):
    try:
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        embedding = twelvelabs_client.embed.create(
            engine_name="Marengo-retrieval-2.6",
            text=text
        ).text_embedding
        return embedding.segments[0].embeddings_float
    except Exception as e:
        st.error(f"Embedding Error: {str(e)}")
        raise e


def get_rag_response(question):
    try:

        question_embedding = emb_text(question)
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=[question_embedding],
            anns_field="vector",
            param=search_params,
            limit=2,
            output_fields=['metadata']
        )

        # Process results with corrected similarity calculation
        retrieved_docs = []
        for hit in results[0]:
            try:
                metadata = hit.entity.metadata
                if metadata:

                    similarity = round((hit.score + 1) * 50, 2)  # Convert from [-1,1] to [0,100]

                    similarity = max(0, min(100, similarity))
                    
                    retrieved_docs.append({
                        "title": metadata.get("title", "Untitled"),
                        "description": metadata.get("description", "No description available"),
                        "product_id": metadata.get("product_id", ""),
                        "video_url": metadata.get("video_url", ""),
                        "link": metadata.get("link", ""),
                        "similarity": similarity,
                        "raw_score": hit.score  # Add raw score for debugging
                    })
                    
                    # # Debug information
                    # st.write(f"Debug - Raw Score: {hit.score}, Calculated Similarity: {similarity}%")
                    
            except Exception as e:
                st.error(f"Error processing hit: {str(e)}")
                continue

        if not retrieved_docs:
            return {
                "response": "I couldn't find any matching products. Try describing what you're looking for differently.",
                "metadata": None
            }

        context = "\n\n".join([f"Title: {doc['title']}\nDescription: {doc['description']}" for doc in retrieved_docs])
        messages = [
            {
                "role": "system",
                "content": """You are a professional fashion advisor and AI shopping assistant.
                Provide stylish, engaging responses about fashion products.
                Focus on style, trends, and helping customers find the perfect items."""
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext: {context}"
            }
        ]

        chat_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        return {
            "response": chat_response.choices[0].message.content,
            "metadata": {
                "sources": retrieved_docs,
                "total_sources": len(retrieved_docs)
            }
        }
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {
            "response": "I encountered an error while processing your request.",
            "metadata": None
        }


def render_product_details(source):
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="product-card">
                <h3 style="color: #FF4B6B;">{source['title']}</h3>
                <div style="margin: 1rem 0;">
                    <div style="background: linear-gradient(90deg, #FF4B6B {source['similarity']}%, #f1f1f1 {source['similarity']}%); 
                         height: 6px; border-radius: 3px; margin-bottom: 0.5rem;"></div>
                    <p style="color: #666;">Similarity Score: {source['similarity']}%</p>
                    <p style="color: #666; font-size: 0.8em;">Raw Score: {source.get('raw_score', 'N/A')}</p>
                </div>
                <p style="color: #333; font-size: 1.1em;">{source['description']}</p>
                <p style="color: #666;">Product ID: {source['product_id']}</p>
                <a href="{source['link']}" target="_blank" style="
                    display: inline-block;
                    background: #FF4B6B;
                    color: white;
                    padding: 0.5rem 1.5rem;
                    border-radius: 25px;
                    text-decoration: none;
                    margin-top: 1rem;
                    transition: all 0.3s ease;
                ">View on Store</a>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if source['video_url']:
                st.video(source['video_url'])

# Main UI
def main():
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #FF4B6B; font-size: 3em; font-weight: 800;">🤵‍♂️ Fashion AI Assistant</h1>
            <p style="color: #666; font-size: 1.2em;">Your personal style advisor powered by AI</p>
        </div>
    """, unsafe_allow_html=True)

    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "👗"):
                if message["role"] == "assistant":
                    st.markdown(message["content"]["response"])
                    if message["content"]["metadata"]:
                        with st.expander("View Product Details 🛍️"):
                            for source in message["content"]["metadata"]["sources"]:
                                render_product_details(source)
                                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about fashion products..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar="🤵‍♂️"):
            with st.spinner("Finding perfect matches..."):
                response_data = get_rag_response(prompt)
                st.markdown(response_data["response"])
                if response_data["metadata"]:
                    with st.expander("View Product Details 🛍️"):
                        for source in response_data["metadata"]["sources"]:
                            render_product_details(source)
                            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response_data})


    with st.sidebar:

        st.markdown("""
        <div style="padding: 1.5rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h2 style="color: #FF4B6B;">Your Fashion Style Guide</h2>
            <p style="color: #666;">How can I help you with, There are various things I can help - </p>
            <ul style="color: #333;">
                <li>Finding perfect outfits</li>
                <li>Style recommendations</li>
                <li>Product information</li>
                <li>Fashion advice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
