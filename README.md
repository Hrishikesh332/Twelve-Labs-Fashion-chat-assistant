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
    <li><a href="#tech-stack">Tech Stack</a></li>
    <li><a href="#instructions-on-running-project-locally">Instructions on Running Project Locally</a></li>
    <li><a href="#feedback">Feedback</a></li>
  </ol>
</details>

------

## About

The Fashion AI Assistant is a recommendation system that combines vector search capabilities with multimodal AI to provide intelligent fashion recommendations. Using TwelveLabs' Marengo 2.6 retrieval embedding model and Milvus vector database, it offers semantic search and personalized fashion suggestions through an intuitive chat interface on streamlit

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

🎯 **Semantic Search**: Advanced search capabilities using TwelveLabs' Marengo 2.6 embeddings

🔍 **AI Powered Chat**: Natural Language Interaction using the LLM (OpenAI model)

🧠 **Smart Recommendation**: Context Aware Fashion Suggestions


I will be updating it the workflow as per the updation -

![](https://github.com/Hrishikesh332/Twelve-Labs-Fashion-chat-assistant/blob/main/src/workflow-fashion-assistant-twelve-labs.png)

## Tech Stack

- **Frontend**: Streamlit, Javascript, CSS
- **Backend**: Streamlit, Python
- **AI Engine**: Integration with Twelve Labs SDK (Marengo 2.6 retreival and Open AI model)
- **Vector Database**: Milvus
- **Deployment**: Streamlit Cloud

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


## Feedback

If you have any feedback, please reach out to us at **hriskikesh.yadav332@gmail.com**
