import streamlit as st
from dotenv import load_dotenv
from utils import get_rag_response, generate_embedding, insert_embeddings, collection

load_dotenv()

st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #fafafa;
    }
    
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e1e1e;
        font-weight: 700;
        padding-bottom: 2rem;
    }
    
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .product-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .nav-button {
        background-color: #81E831 !important;  /* Matching pink from app */
        color: white !important;
        border-radius: 4px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        cursor: pointer !important;
    }
    
    .nav-button:hover {
        background-color: #ff3d60 !important;  /* Slightly darker shade of the same pink */
        opacity: 0.9;
    }
    
    
    .nav-container {
        position: fixed;
        top: 80px;
        right: 20px;
        display: flex;
        gap: 10px;
        z-index: 1000;
    }
    .stButton button {
        background-color: #81E831 !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton button:hover {
        background-color: #eeecec !important;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

def render_product_details(source):
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="product-card">
                <h3 style="color: #81E831;">{source['title']}</h3>
                <div style="margin: 1rem 0;">
                    <div style="background: linear-gradient(90deg, #81E831 {source['similarity']}%, #f1f1f1 {source['similarity']}%); 
                         height: 6px; border-radius: 3px; margin-bottom: 0.5rem;"></div>
                    <p style="color: #666;">Similarity Score: {source['similarity']}%</p>
                </div>
                <p style="color: #333; font-size: 1.1em;">{source['description']}</p>
                <p style="color: #666;">Product ID: {source['product_id']}</p>
                <a href="{source['link']}" target="_blank" style="
                    display: inline-block;
                    background: #81E831;
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

def chat_page():
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #81E831; font-size: 3em; font-weight: 800;">🤵‍♂️ Fashion AI Assistant</h1>
            <p style="color: #666; font-size: 1.2em;">Your personal style advisor powered by AI</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="nav-container">
            <a href="add_product_page" class="nav-button">Add Product Data</a>
            <a href="visual_search" class="nav-button">Visual Search</a>
        </div>
    """, unsafe_allow_html=True)

    chat_container = st.container()
    
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state.messages = []

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

    prompt = st.chat_input("Hey! Ask me anything about fashion - styles, outfits, trends...")
    
    if prompt:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)

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
            <h2 style="color: #81E831;">Your Fashion Style Guide</h2>
            <p style="color: #666;">How can I help you with, There are various things I can help - </p>
            <ul style="color: #333;">
                <li>Finding perfect outfits</li>
                <li>Style recommendations</li>
                <li>Product information</li>
                <li>Fashion advice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    query_params = st.query_params
    page = query_params.get("page", "chat")[0] if query_params.get("page") else "chat"

    if page == "chat":
        chat_page()
    elif page == "add_product":
        add_product_main()
    elif page == "visual_search":
        visual_search_main()

if __name__ == "__main__":
    main()
