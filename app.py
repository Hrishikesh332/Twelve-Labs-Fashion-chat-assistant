import streamlit as st
from dotenv import load_dotenv
import os
from pymilvus import connections, Collection, utility
from openai import OpenAI

# Load environment variables
load_dotenv()
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client and Milvus connection
openai_client = OpenAI()
connections.connect(uri=URL, token=TOKEN)
collection = Collection(COLLECTION_NAME)
collection.load()

def emb_text(text):
    """Generate embeddings using OpenAI's API"""
    result = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
        dimensions=1024
    )
    return result.data[0].embedding

def get_rag_response(question):
    """Get RAG response with metadata"""
    try:
        # Generate embedding and search
        question_embedding = emb_text(question)
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=[question_embedding],
            anns_field="vector",
            param=search_params,
            limit=3,
            output_fields=['metadata']
        )

        # Extract documents and metadata
        retrieved_docs = []
        for hit in results[0]:
            try:
                metadata = hit.entity.metadata
                if metadata:
                    similarity = round((1 - hit.distance) * 100, 2)
                    retrieved_docs.append({
                        "title": metadata.get("title", "Untitled"),
                        "description": metadata.get("description", "No description available"),
                        "product_id": metadata.get("product_id", ""),
                        "video_url": metadata.get("video_url", ""),
                        "link": metadata.get("link", ""),
                        "similarity": similarity
                    })
            except Exception as e:
                continue

        if not retrieved_docs:
            return {
                "response": "I couldn't find any relevant information. Please try another question.",
                "metadata": None
            }

        # Create context from relevant fields
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(f"Title: {doc['title']}\nDescription: {doc['description']}")
        context = "\n\n".join(context_parts)

        # Generate response using ChatGPT
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant for an e-commerce platform. 
                Provide clear, concise answers about products based on the context. 
                Include relevant product details in a natural way."""
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
    """Render product details with embedded video"""
    st.markdown(f"""
    ### {source['title']}
    - **Similarity Score:** {source['similarity']}%
    - **Description:** {source['description']}
    - **Product ID:** {source['product_id']}
    """)
    
    # Product link
    st.markdown(f"**[View Product Page]({source['link']})**")
    
    # Video rendering
    if source['video_url']:
        st.markdown("### Product Video")
        st.video(source['video_url'])

# Streamlit UI
st.title("🛍️ E-commerce Product Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Display main response
            st.markdown(message["content"]["response"])
            
            # Display metadata in expander if available
            if message["content"]["metadata"]:
                with st.expander("View Product Details"):
                    for source in message["content"]["metadata"]["sources"]:
                        render_product_details(source)
                        st.markdown("---")
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about our products..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching products..."):
            response_data = get_rag_response(prompt)
            st.markdown(response_data["response"])
            
            if response_data["metadata"]:
                with st.expander("View Product Details"):
                    for source in response_data["metadata"]["sources"]:
                        render_product_details(source)
                        st.markdown("---")
    
    st.session_state.messages.append({"role": "assistant", "content": response_data})

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    ### 🤖 AI Shopping Assistant
    
    This assistant helps you find products by:
    - Understanding natural language queries
    - Searching through product catalog
    - Providing detailed product information
    - Showing product videos
    
    Ask about any product to get started!
    """)
