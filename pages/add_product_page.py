import streamlit as st
from utils import generate_embedding, insert_embeddings

def add_product_data():
    st.subheader("Add Product Data")
    
    product_id = st.text_input("Product ID")
    title = st.text_input("Title")
    description = st.text_area("Description")
    link = st.text_input("Link")
    video_url = st.text_input("Video URL")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Add Product", type="primary"):
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
    
    with col2:
        st.markdown('<a href="/" class="nav-button">Back to Chat</a>', unsafe_allow_html=True)

def main():
    st.title("Add Product Data")
    add_product_data()

if __name__ == "__main__":
    main()