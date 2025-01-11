import streamlit as st
from utils import generate_embedding, insert_embeddings

def add_product_data():
    st.subheader("Add Product Data")

    col1, col2 = st.columns(2)
    
    with col1:
        product_id = st.text_input("Product ID")
        title = st.text_input("Title")
        description = st.text_area("Description")
    
    with col2:
        link = st.text_input("Link")
        video_url = st.text_input("Video URL")
    
    if st.button("Add Product", type="primary", use_container_width=True):
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
        .nav-button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background-color: #FF4B6B;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Add Product Data")
    add_product_data()

if __name__ == "__main__":
    main()
