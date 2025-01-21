import streamlit as st
from utils import generate_embedding, insert_embeddings

# Set this to False for demonstration mode
ENABLE_INSERTIONS = True  # Change to True to enable insertions

def add_product_data():
    # Add warning for demonstration mode
    if not ENABLE_INSERTIONS:
        st.warning("""
            🚨 **Demo Mode Active**
            
            This is a demonstration version where product insertion is disabled. To use the full functionality:
            1. Fork this project on Replit
            2. Set ENABLE_INSERTIONS to True at the top of this file
            3. Configure your own API keys and environment variables
            
            View the [GitHub Repository](your_repo_link) for setup instructions.
        """)

    col1, col2 = st.columns(2)
    
    with col1:
        product_id = st.text_input("Product ID", disabled=not ENABLE_INSERTIONS)
        title = st.text_input("Title", disabled=not ENABLE_INSERTIONS)
        description = st.text_area("Description", disabled=not ENABLE_INSERTIONS)
    
    with col2:
        link = st.text_input("Link", disabled=not ENABLE_INSERTIONS)
        video_url = st.text_input("Video URL", disabled=not ENABLE_INSERTIONS)
    
    st.markdown(
        """
        <style>
        .custom-button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background-color: #81E831;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 1rem;
            width: 100%;
            text-align: center;
            cursor: pointer;
            opacity: var(--button-opacity);
        }
        .custom-button.disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    button_class = "custom-button" + (" disabled" if not ENABLE_INSERTIONS else "")
    if st.markdown(f'<div class="{button_class}">Insert Product</div>', unsafe_allow_html=True):
        if not ENABLE_INSERTIONS:
            st.info("Product insertion is disabled in demonstration mode.")
            return
            
        if product_id and title and description and link and video_url:
            product_data = {
                "product_id": product_id,
                "title": title,
                "desc": description,
                "link": link,
                "video_url": video_url
            }
            
            with st.spinner("Processing product..."):
                embeddings, error = generate_embedding(product_data)
                
                if error:
                    st.error(f"Error processing product: {error}")
                else:
                    insert_result = insert_embeddings(embeddings, product_data)
                    
                    if insert_result:
                        st.success("Product data added successfully!")
                    else:
                        st.error("Failed to add product data.")
        else:
            st.warning("Please fill in all fields.")
    
    st.markdown('<a href="/" class="nav-button">Back to Chat</a>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Add Product Data", page_icon=":package:")
    st.markdown(
        """
        <style>
        .header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #81E831;
            margin-bottom: 1rem;
            text-align: center;
        }
        .nav-button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background-color: #81E831;
            color: white !important;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 1rem;
        }
        .nav-button:hover {
            color: white !important;
            text-decoration: none;
        }
        
        /* Add styles for demo mode warning */
        .demo-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            border-radius: 4px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="header">Product Data Catalogue</div>', unsafe_allow_html=True)
    st.title("Insert Product Data")
    add_product_data()

if __name__ == "__main__":
    main()
