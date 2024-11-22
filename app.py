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
        
        # First, let's try to query and see what fields are available
        results = collection.search(
            data=[question_embedding],
            anns_field="vector",
            param=search_params,
            limit=3,
            output_fields=['*']  # Try to get all fields
        )

        # Debug: Print first result structure
        if results and len(results) > 0 and len(results[0]) > 0:
            first_hit = results[0][0]
            st.write("First hit structure:", first_hit.to_dict())
            st.write("Entity fields:", vars(first_hit.entity))

        # Extract documents and metadata
        retrieved_docs = []
        for hit in results[0]:
            try:
                # Try different ways to access the text content
                doc_dict = hit.to_dict()
                entity_vars = vars(hit.entity)
                
                content = None
                # Try different possible locations of the text
                if hasattr(hit.entity, 'text_field'):
                    content = hit.entity.text_field
                elif hasattr(hit.entity, 'text'):
                    content = hit.entity.text
                elif '$meta' in entity_vars and 'text' in entity_vars['$meta']:
                    content = entity_vars['$meta']['text']
                elif 'text' in doc_dict:
                    content = doc_dict['text']
                
                if content:
                    similarity = round((1 - hit.distance) * 100, 2)
                    retrieved_docs.append({
                        "content": content,
                        "similarity": similarity
                    })
                else:
                    st.write("No content found in hit:", doc_dict)
            except Exception as e:
                st.error(f"Error processing hit: {str(e)}")
                continue

        if not retrieved_docs:
            return {
                "response": "Unable to retrieve information. Please verify the data structure in Milvus.",
                "metadata": None
            }

        # Format context
        context = "\n\n".join([f"{i+1}. {doc['content']}" for i, doc in enumerate(retrieved_docs)])

        # Generate response using ChatGPT
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that provides clear, accurate answers about Milvus based on the given context. Use markdown formatting for better readability."
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
        st.error(f"Debug - Error details: {str(e)}")
        st.write("Collection Schema:", collection.schema)
        return {
            "response": "Error accessing the knowledge base. Please check the debug information.",
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
            st.markdown(message["content"]["response"])
            if message["content"]["metadata"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["content"]["metadata"]["sources"], 1):
                        st.markdown(f"""
                        **Source {i}** (Similarity: {source['similarity']}%)
                        ```
                        {source['content']}
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
        with st.spinner("Searching knowledge base..."):
            response_data = get_rag_response(prompt)
            st.markdown(response_data["response"])
            
            if response_data["metadata"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(response_data["metadata"]["sources"], 1):
                        st.markdown(f"""
                        **Source {i}** (Similarity: {source['similarity']}%)
                        ```
                        {source['content']}
                        ```
                        """)
    
    st.session_state.messages.append({"role": "assistant", "content": response_data})

# Sidebar info
with st.sidebar:
    st.title("About")
    st.markdown("""
    ### 🤖 AI-Powered Milvus Chat
    
    This chatbot uses:
    - Semantic search with Milvus
    - OpenAI embeddings & chat
    - Source verification
    
    Ask questions about Milvus documentation!
    """)

    # Show collection stats
    with st.expander("Collection Info"):
        st.write("Schema:", collection.schema)
        try:
            st.write("Size:", collection.num_entities)
        except:
            st.write("Size: Unknown")
