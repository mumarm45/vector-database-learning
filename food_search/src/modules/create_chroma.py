import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from modules.load_food_data import load_food_data, populate_similarity_collection
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def create_chroma_client():
    return chromadb.Client()

def add_to_collection(collection, documents, metadatas, ids):
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )   
    return collection 

def create_similarity_search_collection(collection_name: str, collection_metadata: dict = None):
    """Create ChromaDB collection with sentence transformer embeddings"""
    client = create_chroma_client()
    try:
        # Try to delete existing collection to start fresh
        client.delete_collection(collection_name)
    except:
        pass
    
    # Create embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create new collection
    return client.create_collection(
        name=collection_name,
        metadata=collection_metadata,
        configuration={
            "hnsw": {"space": "cosine"},
            "embedding_function": sentence_transformer_ef
        }
    )

def initiate_food_collection(): 
    """Initiate food collection with ChromaDB"""
    file_path = os.path.join(root_path, "FoodDataSet.json")  
    food_items = load_food_data(file_path)
    collection = create_similarity_search_collection("food_collection")
    documents, metadatas, ids = populate_similarity_collection(collection, food_items)
    add_to_collection(collection, documents, metadatas, ids)
    return collection