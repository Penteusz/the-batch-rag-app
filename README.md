# The Batch RAG

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that processes articles and images from "The Batch" newsletter by DeepLearning.AI. It combines web scraping, document processing, and image analysis to create a searchable knowledge base with both text and image content.

## Features

1. Web scraping of articles from deeplearning.ai
2. Image extraction and captioning using AI models
3. Document vectorization and semantic search
4. Combined text and image querying capabilities
5. Streamlit-based user interface
6. Comprehensive evaluation system with multiple metrics

## Project Structure

```plaintext
RAG_APP/
├── evaluation/                            # Evaluation modules
│   ├── build_dataset.py                   # Dataset management
│   ├── correctness.py                     # Correctness evaluation
│   ├── groundedness.py                    # Groundedness evaluation
│   ├── relevance.py                       # Relevance evaluation
│   └── retrieval_relevance.py             # Retrieval relevance evaluation
├── executors/                             # Main execution scripts
│   ├── execute_rag_preprocessing.py       # Data preprocessing
│   └── execute_evaluation.py              # System evaluation
├── frontend/                              # User interface
│   ├── app.py                             # Frontend app
│   └── requirements.txt                   # System evaluationFrontend app dependencies
├── helpers/                               # Utility functions
│   ├── ingestion.py                       # Common ingestion utilities
│   ├── logger_config.py                   # Logging configuration
│   └── vectorstore_manager.py             # Vector store operations
├── ingestion/                             # Data collection
│   ├── scrape_images.py                   # Image scraping
│   └── sitemap_scraper.py                 # Article URL extraction
├── logs/                                  # Application logs
├── preprocessing/                         # Data processing
│   ├── documents_processing.py            # Document vectorization
│   └── image_preprocess.py                # Image captioning
├── config.py                              # Configuration settings
├── requirements.txt                       # Project dependencies
├── Dockerfile                             # Container configuration
├── .dockerignore                          # Docker build exclusions
└── .env                                   # Environment variables
```

## Prerequisites

1. Python 3.10 or higher
2. OpenAI API key
3. HuggingFace Hub API token
4. LangSmith API key (for evaluation)

## Setup

1. **Create Project Directory (e.g., The_Batch_RAG)**
   ```bash
   mkdir The_Batch_RAG
   cd The_Batch_RAG
   ```

2. **Clone the Repository**
   ```bash
   git clone [<repository-url>](https://github.com/Penteusz/the-batch-rag-app.git)
   cd the-batch-rag-app
   ```

3. **Create and Activate Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/MacOS
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```plaintext
   # Required
   OPENAI_API_KEY="your_openai_api_key_here"             # OpenAI API key for embeddings and chat
   HUGGINGFACEHUB_API_TOKEN="your_token_here"            # For image captioning model

   # Optional (for evaluation)
   LANGSMITH_API_KEY="your_langsmith_api_key_here"       # For evaluation metrics
   LANGSMITH_PROJECT="the-batch-rag-app"                 # Project name in LangSmith
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com    # LangSmith endpoint
   LANGSMITH_TRACING=true                                # Enable tracing for debugging
   ```

## Configuration Settings

The `config.py` file contains important settings that control the behavior of the application:

1. **Data Processing Limits**
   - `TEST_RUN`: When True, runs the application in test mode with limited data (default: True)
   - `LOADED_ARTICLES_LIMIT`: Maximum number of articles to process when TEST_RUN is True (None for all, default: 100)
   - `LOADED_IMAGES_LIMIT`: Maximum number of images to process when TEST_RUN is True (None for all, default: 50)
   - `BATCH_SIZE`: Number of items to process in each batch (default: 50)
   - `BATCH_LIMIT`: Maximum number of batches to process when TEST_RUN is True (None for all)

2. **LLM Settings**
   - `OPENAI_MODEL_NAME`: The OpenAI model to use (default: "gpt-3.5-turbo")
   - `MODEL_TEMPERATURE`: Controls randomness in responses (default: 0.0)
   - `TOKEN_LIMIT`: Maximum tokens for context window (default: 12000)
   - `CHATOPENAI_MAX_TOKENS`: Maximum tokens in response (default: 1024)

3. **Image Processing**
   - `IMAGE_TO_TEXT_MODEL`: Model for image captioning (default: "Salesforce/blip-image-captioning-large")
   - Note: GPU is used if available for faster image captioning

4. **Retrieval Settings**
   - `RETRIEVED_DOCS_COUNT`: Number of documents to retrieve (default: 5)
   - `VECTORSTORE_PATH`: Location of the FAISS index


## Executing ingestion and preprocessing

```bash
python executors/execute_rag_preprocessing.py
```

This script will:
- Scrape article URLs from The Batch newsletter
- Download and process images
- Generate image captions
- Create vector embeddings
- Store everything in a vector database
<br>

## Evaluate the RAG system's performance   
   ```bash
   python executors/execute_evaluation.py
   ```
This will assess the system on multiple metrics:
   - Correctness: Measures factual accuracy of generated answers. Compares against ground truth responses
   - Groundedness: Evaluates if answers are supported by source documents. Checks for hallucinations
   - Relevance: Assesses how well answers address the questions. Considers both content and context
   - Retrieval Relevance: Evaluates quality of retrieved documents. Measures relevance to query

## Running the Application

   ```bash
   streamlit run frontend/app.py
   ```

**Aternatively, you can run the app in a Docker container:**
```bash
docker build -t rag-app:v1 .

docker run -p 8505:8505 -e OPENAI_API_KEY="your-openai-api-key" rag-app:v1
```