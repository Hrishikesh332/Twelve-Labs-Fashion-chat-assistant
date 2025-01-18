import os
import uuid
from dotenv import load_dotenv
from twelvelabs import TwelveLabs
from pymilvus import connections, Collection
import streamlit as st
from openai import OpenAI
import numpy as np

load_dotenv()

# Load environment variables
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')
TWELVELABS_API_KEY = os.getenv('TWELVELABS_API_KEY')

# Initialize connections
openai_client = OpenAI()
connections.connect(uri=URL, token=TOKEN)
collection = Collection(COLLECTION_NAME)
collection.load()


# Generate text and segmented video embeddings for a product
def generate_embedding(product_info):
    try:
        st.write("Starting embedding generation process...")
        st.write(f"Processing product: {product_info['title']}")
        
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        st.write("TwelveLabs client initialized successfully")
        
        st.write("Attempting to generate text embedding...")

        text = f"product type: {product_info['title']}. " \
               f"product description: {product_info['desc']}. " \
               f"product category: fashion apparel."
               
        st.write(f"Generating embedding for text: {text}")
        
        text_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=text
        ).text_embedding.segments[0].embeddings_float
        st.write("Text embedding generated successfully")

        
        # Create and wait for video embedding task
        st.write("Creating video embedding task...")
        video_task = twelvelabs_client.embed.task.create(
            model_name="Marengo-retrieval-2.7",
            video_url=product_info['video_url'],
            video_clip_length=6
        )
        
        def on_task_update(task):
            st.write(f"Video processing status: {task.status}")
        
        st.write("Waiting for video processing to complete...")
        video_task.wait_for_done(sleep_interval=2, callback=on_task_update)
        
        # Retrieve segmented video embeddings
        video_task = video_task.retrieve()
        if not video_task.video_embedding or not video_task.video_embedding.segments:
            raise Exception("Failed to retrieve video embeddings")
        
        video_segments = video_task.video_embedding.segments
        st.write(f"Retrieved {len(video_segments)} video segments")
        
        video_embeddings = []
        for segment in video_segments:
            video_embeddings.append({
                'embedding': segment.embeddings_float,
                'metadata': {
                    'scope': 'clip',
                    'start_time': segment.start_offset_sec,
                    'end_time': segment.end_offset_sec,
                    'video_url': product_info['video_url']
                }
            })
        
        return {
            'text_embedding': text_embedding,
            'video_embeddings': video_embeddings
        }, None
        
    except Exception as e:
        st.error("Error in embedding generation")
        st.error(f"Error message: {str(e)}")
        return None, str(e)


# Insert text and all video segment embeddings
def insert_embeddings(embeddings_data, product_info):
    try:
        metadata = {
            "product_id": product_info['product_id'],
            "title": product_info['title'],
            "description": product_info['desc'],
            "video_url": product_info['video_url'],
            "link": product_info['link']
        }
        
        # Insert text embedding
        text_entry = {
            "id": int(uuid.uuid4().int & (1<<63)-1),
            "vector": embeddings_data['text_embedding'],
            "metadata": metadata,
            "embedding_type": "text"
        }
        collection.insert([text_entry])
        st.write("Text embedding inserted successfully")
        
        # Insert each video segment embedding
        for video_segment in embeddings_data['video_embeddings']:
            video_entry = {
                "id": int(uuid.uuid4().int & (1<<63)-1),
                "vector": video_segment['embedding'],
                "metadata": {**metadata, **video_segment['metadata']},
                "embedding_type": "video"
            }
            collection.insert([video_entry])
        
        st.write(f"Inserted {len(embeddings_data['video_embeddings'])} video segment embeddings")
        return True
        
    except Exception as e:
        st.error(f"Error inserting embeddings: {str(e)}")
        return False


# Search for similar video segments using image query
def search_similar_videos(image_file, top_k=5):
    
    try:
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        image_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            image_file=image_file
        ).image_embedding.segments[0].embeddings_float
        
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 1024,
                "ef": 64
            }
        }
        
        results = collection.search(
            data=[image_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr="embedding_type == 'video'",
            output_fields=["metadata"]
        )

        search_results = []
        for hits in results:
            for hit in hits:
                metadata = hit.metadata
                # Convert score from [-1,1] to [0,100] range
                similarity = round((hit.score + 1) * 50, 2)
                similarity = max(0, min(100, similarity))
                
                search_results.append({
                    'Title': metadata.get('title', ''),
                    'Description': metadata.get('description', ''),
                    'Link': metadata.get('link', ''),
                    'Start Time': f"{metadata.get('start_time', 0):.1f}s",
                    'End Time': f"{metadata.get('end_time', 0):.1f}s",
                    'Video URL': metadata.get('video_url', ''),
                    'Similarity': f"{similarity}%",
                    'Raw Score': hit.score
                })
        
        # Sort by similarity score in descending order
        search_results.sort(key=lambda x: float(x['Similarity'].rstrip('%')), reverse=True)
        
        return search_results
        
    except Exception as e:
        return None


# Get response using text embeddings to get multimodal result
def get_rag_response(question):
    try:
        # Initialize TwelveLabs client
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        
        # Generate embedding for the question with fashion context
        question_with_context = f"fashion product: {question}"
        question_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=question_with_context
        ).text_embedding.segments[0].embeddings_float
        
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 1024,
                "ef": 64
            }
        }
        
        # Search for relevant text embeddings
        text_results = collection.search(
            data=[question_embedding],
            anns_field="vector",
            param=search_params,
            limit=2,  # Get top 2 text matches
            expr="embedding_type == 'text'",
            output_fields=["metadata"]
        )
        
        # Search for relevant video segments
        video_results = collection.search(
            data=[question_embedding],
            anns_field="vector",
            param=search_params,
            limit=3,  # Get top 3 video segments
            expr="embedding_type == 'video'",
            output_fields=["metadata"]
        )

        # Process text results
        text_docs = []
        for hits in text_results:
            for hit in hits:
                metadata = hit.metadata
                similarity = round((hit.score + 1) * 50, 2)
                similarity = max(0, min(100, similarity))
                
                text_docs.append({
                    "title": metadata.get('title', 'Untitled'),
                    "description": metadata.get('description', 'No description available'),
                    "product_id": metadata.get('product_id', ''),
                    "video_url": metadata.get('video_url', ''),
                    "link": metadata.get('link', ''),
                    "similarity": similarity,
                    "raw_score": hit.score,
                    "type": "text"
                })

        # Process video results
        video_docs = []
        for hits in video_results:
            for hit in hits:
                metadata = hit.metadata
                similarity = round((hit.score + 1) * 50, 2)
                similarity = max(0, min(100, similarity))
                
                video_docs.append({
                    "title": metadata.get('title', 'Untitled'),
                    "description": metadata.get('description', 'No description available'),
                    "product_id": metadata.get('product_id', ''),
                    "video_url": metadata.get('video_url', ''),
                    "link": metadata.get('link', ''),
                    "similarity": similarity,
                    "raw_score": hit.score,
                    "start_time": metadata.get('start_time', 0),
                    "end_time": metadata.get('end_time', 0),
                    "type": "video"
                })

        if not text_docs and not video_docs:
            return {
                "response": "I couldn't find any matching products. Try describing what you're looking for differently.",
                "metadata": None
            }

        # Create context from text results only for LLM
        text_context = "\n\n".join([
            f"Product: {doc['title']}\nDescription: {doc['description']}\nLink: {doc['link']}"
            for doc in text_docs
        ])

        # Create messages for chat completion
        messages = [
            {
                "role": "system",
                "content": """You are a professional fashion advisor and AI shopping assistant.
                Organize your response in the following format:

                First, provide a brief, direct answer to the user's query
                Then, describe any relevant products found that match their request, including:
                   - Product name and key features
                   - Why this product matches their needs
                   - Style suggestions for how to wear or use the item
                Finally, provide any additional style advice or recommendations
                
                Keep your response engaging and natural while maintaining this clear structure.
                Focus on being helpful and specific rather than promotional."""
            },
            {
                "role": "user",
                "content": f"""Query: {question}

Available Products:
{text_context}

Please provide fashion advice and product recommendations based on these options."""
            }
        ]

        # Get response from OpenAI
        chat_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        # Format and return response
        return {
            "response": chat_response.choices[0].message.content,
            "metadata": {
                "sources": text_docs + video_docs,
                "total_sources": len(text_docs) + len(video_docs),
                "text_sources": len(text_docs),
                "video_sources": len(video_docs)
            }
        }
    
    except Exception as e:
        st.error(f"Error in multimodal RAG: {str(e)}")
        return {
            "response": "I encountered an error while processing your request. Please try again.",
            "metadata": None
        }
 

# Extract video ID and platform from URL
def get_video_id_from_url(video_url):

    try:
        if 'vimeo.com' in video_url:
            video_id = video_url.split('/')[-1].split('?')[0]
            return video_id, 'vimeo'
        else:
            return video_url, 'direct'
    except Exception as e:
        st.error(f"Error processing video URL: {str(e)}")
        return None, None

# Format time in seconds to URL compatible format
def format_time_for_url(time_in_seconds):
    try:
        return str(int(float(time_in_seconds)))
    except:
        return "0"

def create_video_embed(video_url, start_time=0, end_time=0):
    try:
        video_id, platform = get_video_id_from_url(video_url)
        start_seconds = format_time_for_url(start_time)
        
        if not video_id:
            return f"<p>Unable to process video URL: {video_url}</p>"
        
        if platform == 'vimeo':
            return f"""
                <iframe 
                    width="100%" 
                    height="315" 
                    src="https://player.vimeo.com/video/{video_id}?autoplay=0#t={start_seconds}s"
                    frameborder="0" 
                    allow="fullscreen; picture-in-picture" 
                    allowfullscreen>
                </iframe>
            """
        elif platform == 'direct':
            return f"""
                <video 
                    width="100%" 
                    height="315" 
                    controls 
                    preload="metadata"
                    id="video-player">
                    <source src="{video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <script>
                    document.getElementById('video-player').addEventListener('loadedmetadata', function() {{
                        this.currentTime = {start_time};
                        this.pause();
                    }});
                </script>
            """
        else:
            return f"<p>Unsupported video platform for URL: {video_url}</p>"
            
    except Exception as e:
        st.error(f"Error creating video embed: {str(e)}")
        return f"<p>Error creating video embed for URL: {video_url}</p>"
