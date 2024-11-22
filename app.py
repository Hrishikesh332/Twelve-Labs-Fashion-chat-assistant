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

# Initialize OpenAI client
openai_client = OpenAI()

# Milvus Connection
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
        # Generate embedding
        question_embedding = emb_text(question)
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        # Search in Milvus
        results = collection.search(
            data=[question_embedding],
            anns_field="vector",
            param=search_params,
            limit=3,
            output_fields=["content"]  # Assuming 'content' is the field name
        )

        # Extract documents and metadata
        retrieved_docs = []
        for hit in results[0]:
            if hasattr(hit.entity, 'content'):
                content = hit.entity.content
                similarity = round((1 - hit.distance) * 100, 2)  # Convert distance to similarity percentage
                retrieved_docs.append({
                    "content": content,
                    "similarity": similarity
                })

        if not retrieved_docs:
            return {
                "response": "No relevant information found.",
                "metadata": None
            }

        # Format context with similarity scores
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Document {i} - Similarity: {doc['similarity']}%]\n{doc['content']}\n")
        
        context = "\n".join(context_parts)

        # Generate response using ChatGPT
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that provides clear, accurate answers about Milvus based on the given context. Format your responses using markdown for better readability."
            },
            {
                "role": "user",
                "content": f"""Based on the following context, answer this question: {question}

Context:
{context}"""
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
        return {
            "response": f"Error: {str(e)}",
            "metadata": None
        }

# Streamlit UI
st.title("📚 Milvus RAG Chatbot")

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
                with st.expander("View Source Documents"):
                    for i, source in enumerate(message["content"]["metadata"]["sources"], 1):
                        st.markdown(f"""
                        **Document {i}**
                        - Similarity Score: {source['similarity']}%
                        ```
                        {source['content'][:200]}...
                        ```
                        """)
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Milvus..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching and analyzing..."):
            response_data = get_rag_response(prompt)
            st.markdown(response_data["response"])
            
            if response_data["metadata"]:
                with st.expander("View Source Documents"):
                    for i, source in enumerate(response_data["metadata"]["sources"], 1):
                        st.markdown(f"""
                        **Document {i}**
                        - Similarity Score: {source['similarity']}%
                        ```
                        {source['content'][:200]}...
                        ```
                        """)
    
    st.session_state.messages.append({"role": "assistant", "content": response_data})

# Sidebar info
with st.sidebar:
    st.title("About")
    st.markdown("""
    ### Features
    - Real-time semantic search
    - Source document tracking
    - Similarity scoring
    - Markdown-formatted responses
    
    ### Technologies
    - 🔍 Milvus Vector DB
    - 🤖 OpenAI Embeddings & Chat
    - 🚀 Streamlit UI
    """)
