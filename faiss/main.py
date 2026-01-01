import re
import numpy as np
import tensorflow_hub as hub
import faiss
from sklearn.datasets import fetch_20newsgroups



def load_newsgroups():
    """Load 20 newsgroups dataset with retry logic"""
    try:
        return fetch_20newsgroups(subset='train')
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Using sample data instead...")
        return None

def preprocess_text(text):
    # Remove email headers
    text = re.sub(r'^From:.*\n?', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove punctuations and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def embed_text_func(texts, embed_model):
    """Embed texts - expects a list of strings"""
    if isinstance(texts, str):
        texts = [texts]
    return embed_model(texts).numpy()

def search(query_text, preprocess_func,embed_text, k=5):
    # Preprocess the query text
    preprocessed_query = preprocess_func(query_text)
    # Generate the query vector
    query_vector = embed_text([preprocessed_query])
    # Perform the search
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances, indices
    
def main():
    print("Hello from faiss!")
    
    # newsgroups_train = load_newsgroups()
    data = [
            "From: user@example.com\nSubject: Space exploration\nNASA announced new missions to Mars.",
            "From: dev@tech.com\nSubject: Programming tips\nPython is great for machine learning.",
            "From: sports@news.com\nSubject: Baseball season\nThe playoffs are starting next week.",
        ]
    
    
    processed_documents = [preprocess_text(doc) for doc in data]
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    X_use = np.vstack([embed_text_func(processed_documents, embed)])
    dimension = X_use.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Creating a FAISS index
    index.add(X_use)  # Adding the document vectors to the index
    
    preprocessed_query = preprocess_text("Python programming tips")
    
    # Test search
    query_vector = embed_text_func([preprocessed_query], embed)
    distances, indices = index.search(query_vector.astype('float32'), 3)
    print(f"Search results indices: {indices[0]}")
    print(f"Search distances: {distances[0]}")
    
    for i, idx in enumerate(indices[0]):
        # Displaying the original (unprocessed) document corresponding to the search result
        print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{data[idx]}\n")    

if __name__ == "__main__":
    main()
