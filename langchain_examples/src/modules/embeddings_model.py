from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from .splliter import text_splitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from .llm_model import llm_model_langchain
import logging
from PyPDF2 import PdfReader
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)
    
## Text Document Retriever

def text_document_retriever(k: int = 5):
    print(f"root_path: {root_path}")
    documents = TextLoader(os.path.join(root_path, "companypolicies.txt")).load()

    documents = text_splitter(documents, chunk_size=2000, chunk_overlap=100)
    vectordb = Chroma.from_documents(documents, embeddings_model())
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    if not documents:
        raise ValueError("No pages were loaded from the PDF")


    return retriever


## PDF Document Retriever and MultiQueryRetriever
def pdf_document_retriever(k: int = 5):
    print(f"root_path: {root_path}")
    documents = PyPDFLoader(os.path.join(root_path, "langchain-paper.pdf")).load()

    documents = text_splitter(documents, chunk_size=500, chunk_overlap=20)
    vectordb = Chroma.from_documents(documents, embeddings_model())
    retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm_model_langchain()
)
    
    if not documents:
        raise ValueError("No pages were loaded from the PDF")


    return retriever


    

def call_retriever(query, k: int = 5):
    retriever = pdf_document_retriever(k)
    return retriever.invoke(query)
    