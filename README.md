# Towards Data Science Article Search API
This project builds a simple semantic search API for articles from the Towards Data Science website, using Sentence Transformers, FAISS for efficient similarity search, and FastAPI for serving a search endpoint.
## About
- Dataset: Articles crawled from the Towards Data Science website.
- Embedding Model: all-MiniLM-L6-v2 from Sentence Transformers.
- Vector Search Engine: FAISS (Facebook AI Similarity Search).
- API Server: FastAPI.
This API allows you to query the dataset and retrieve the most semantically similar articles.
## How It Works
1. Load Dataset: Articles are loaded and preprocessed from tds_articles_content.xlsx.
2. Embedding: Each article is encoded into a dense vector using a pre-trained model.
3. FAISS Index: Embeddings are indexed using FAISS for fast nearest-neighbor search.
4. FastAPI: An API endpoint /search is provided to send search queries and receive relevant articles.
