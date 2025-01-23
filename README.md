<br />
<div align="center">
  <h3 align="center">Fashion AI Assistant with Twelve Labs</h3>
  <p align="center">
    Power of Search with the Twelve Labs Embed API and Milvus
    <br />
    <a href="https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant">Explore the docs »</a>
    <br />
    <br />
    <a href="https://fashion-chat-twelvelabs.streamlit.app/">View Demo</a> ·
    <a href="https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant/issues">Report Bug</a> ·
    <a href="https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant/issues">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#demonstration">Demonstration</a></li>
    <li><a href="#workflow">Workflow</a></li>
    <li><a href="#tech-stack">Tech Stack</a></li>
    <li><a href="#instructions-on-running-project-locally">Instructions on Running Project Locally</a></li>
    <li><a href="#usecases">Usecase</a></li>
    <li><a href="#feedback">Feedback</a></li>
  </ol>
</details>

------

## About

Discover your perfect style with Fashion AI Assistant! This application combines the power of visual search, conversational AI, and video understanding to innovate how you explore and find fashion. Whether you're chatting about your style preferences or sharing a photo of an outfit you want, the application helps you discover exactly what you're looking for ✨

Built with TwelveLabs marengo-retrieval-2.7 model, Milvus vector database, and OpenAI's gpt-3.5, this application brings together the latest in AI technology to create a personalized shopping experience. From finding similar products in video content to providing tailored fashion advice 🛍️

## Demonstration

Try the Application Now:

<a href="https://fashion-ai-chat.streamlit.app" target="_blank" style="
    display: inline-block;
    padding: 12px 24px;
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
    background-color: #007bff;
    border: none;
    border-radius: 8px;
    text-align: center;
    text-decoration: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transition: background-color 0.3s, box-shadow 0.3s;
">
    Fashion AI Assistant Demo
</a>



## Features

🤖 Multimodal Search: Seamlessly search through fashion products using both text descriptions and image queries powered by TwelveLabs marengo-retrieval-2.7 model for embedding generation and the Milvus for vector database.

🎯 Visual Product Discovery: Upload images to find similar products and see exact video segments where they appear, with precise timestamps.

💬 AI Fashion Assistant: Natural conversation with the help of chatbot about style preferences and receive personalized fashion recommendations using gpt-3.5.

💡 **Smart Suggestions**: Helpful prompt suggestions to guide users in discovering fashion products and styles effectively.


## Demonstration

Demo #1 - Insertion of Product Catalogue into the Milvus Collection

![](https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant/blob/main/src/demo_fashion_insertion.gif)


Demo #2 - In this example, the product image - Black shirt - is provided as a query; the result is a video segment with metadata for this product.

![](https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant/blob/main/src/demo_visual_search.gif)


Demo #3 - This example provides the query with the suggestion - "I'm looking for a black T-shirt", and LLM provides the result with the suggestions on styling and the product, and also the video segments.

![](https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant/blob/main/src/demo_rag_1.gif)

Demo #4 - The following example provides the query "Suggest the Indian bridal wear", then it provides the relevant information around the various data modalities.

![](https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant/blob/main/src/demo_rag_2.gif)

## Workflow

#1 Multimodal Retreival Augment Generation Conversation Flow in App

![Multimodal Retreival Augment Generation Conversation Flow in App](https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant/blob/main/src/Chat_RAG_flow.png)

#2 Semantic Search from Image to Video Segments

![Semantic Search from Image to Video Segments](https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant/blob/main/src/semantic_similar_videos_architecture.png)


## Tech Stack

- **Frontend**: Streamlit, Javascript, CSS
- **Backend**: Streamlit, Python
- **AI Engine**: Integration with Twelve Labs SDK (Marengo 2.7 retreival and Open AI model)
- **Vector Database**: Milvus
- **Deployment**: Streamlit Cloud

Replit Repo Link - [Fashion AI Assistant Template](https://replit.com/@twelvelabs/Twelve-Labs-Fashion-chat-assistant?v=1)

## Instructions on Running Project Locally

To run the **Fashion AI Assistant** locally, follow these steps -

### Step 1 - Clone the Project

```bash
git clone https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant.git
```

Install Dependencies

```
 cd Twelve-Labs-Fashion-chat-assistant
 
 pip install -r requirements.txt
```

Prepare the .env file as per the instrcution. The .env file is provided below

```
TWELVELABS_API_KEY="your_twelvelabs_key"
COLLECTION_NAME="your_collection_name"
URL="your_milvus_url"
TOKEN="your_milvus_token"
OPENAI_API_KEY="your_openai_key"
```

To Run the Server Locally

```
python app.py
```

The application is live at -

```
http://localhost:8501/
```

## Usecases


🛍️ E-commerce: Enhance product search and recommendations using text and image queries.

🎵 Music Discovery: Find similar songs, artists, or genres based on audio clips and user preferences

🎥 Intelligent Video Search Engine: Retrieves videos based on visual and audio appearing in the content. Enables efficient search of video for content creators, journalists, and researchers

🗺️ Personalized Travel Planner: Curates travel itineraries based on user preferences, reviews, and destination data.

📚Educational Resource Management: Organize and retrieve learning materials, such as text documents, presentations, videos, and interactive simulations, based on content and pedagogical requirements.

🏀Sports Analytics: Analyze player and team performance using a combination of video footage, sensor data, and statistical records to inform coaching decisions and strategies.


## Feedback

If you have any feedback, please reach out to us at **hriskikesh.yadav332@gmail.com**
