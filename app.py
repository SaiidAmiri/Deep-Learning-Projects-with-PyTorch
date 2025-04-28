import pandas as pd
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load a Pretrained Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Data Preprocessing
df = pd.read_excel("tds_articles_content.xlsx")    # The dataset includes articles crawled from the Towards Data Science website.
articles = df["article"].dropna().drop_duplicates().tolist()

# Convert Articles to Vector Embeddings
embeddings = model.encode(articles)

# Store Embeddings in a FAISS Index
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))  # Store embeddings

# Check the dimension of vectors in the FAISS index
print("Vector dimension:", index.d)

# Check the number of vectors stored in the FAISS index
print("Number of stored vectors:", index.ntotal)

# FastAPI Setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Function to Perform Search
def search_articles(query, top_k=3):
    query_embedding = model.encode([query])  # Convert query to vector
    _, indices = index.search(np.array(query_embedding), top_k)  # FAISS similarity search
    results = [articles[i] for i in indices[0]]  # Retrieve top-k articles
    return results

@app.post("/search")
def search(request: QueryRequest):
    results = search_articles(request.query, top_k=3)
    return {"results": results}