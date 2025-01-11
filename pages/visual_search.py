import streamlit as st
from utils import search_similar_videos, create_video_embed

def main():
    st.set_page_config(page_title="Visual Search", page_icon=":mag:")
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
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="header">Visual Search</div>', unsafe_allow_html=True)
    st.subheader("Search Similar Product Clips")
    
    st.markdown('<a href="/" class="nav-button">Back to Chat</a>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg'],
                help="Select an image to find similar video segments"
            )
            
            if uploaded_file:
                st.image(uploaded_file, caption="Query Image", use_column_width=True)
        
        with col2:
            if uploaded_file:
                st.subheader("Search Parameters")
                top_k = st.slider(
                    "Number of results",
                    min_value=1,
                    max_value=20,
                    value=2,
                    help="Select the number of similar videos to retrieve"
                )
                
                if st.button("Search", type="primary", use_container_width=True):
                    with st.spinner("Searching for similar videos..."):
                        results = search_similar_videos(uploaded_file, top_k=top_k)
                        
                        if not results:
                            st.warning("No similar videos found")
                        else:
                            st.subheader("Results")
                            for idx, result in enumerate(results, 1):
                                with st.expander(f"Match #{idx} - Similarity: {result['Similarity']}", expanded=(idx==1)):
                                    video_col, details_col = st.columns([2, 1])
                                    
                                    with video_col:
                                        st.markdown("#### Video Segment")
                                        video_embed = create_video_embed(
                                            result['Video URL'],
                                            float(result['Start Time'].replace('s', '')),
                                            float(result['End Time'].replace('s', ''))
                                        )
                                        st.markdown(video_embed, unsafe_allow_html=True)
                                    
                                    with details_col:
                                        st.markdown(f"""
                                            #### Details
                                            
                                            📝 **Title**  
                                            {result['Title']}
                                            
                                            📖 **Description**  
                                            {result['Description']}
                                            
                                            🔗 **Link**  
                                            [Open Product]({result['Link']})
                                            
                                            🕒 **Time Range**  
                                            {result['Start Time']} - {result['End Time']}
                                            
                                            📊 **Similarity Score**  
                                            {result['Similarity']}
                                        """)

if __name__ == "__main__":
    main()
