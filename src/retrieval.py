import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time

@st.cache_resource
def get_embeddings():
    """Returns the standardized embedding model using langchain-huggingface."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_pinecone_client(api_key):
    """Returns a cached Pinecone client."""
    return Pinecone(api_key=api_key)

@st.cache_resource
def get_vectorstore(_embeddings, pinecone_api_key, index_name):
    """Returns a cached PineconeVectorStore instance."""
    return PineconeVectorStore(
        index_name=index_name,
        embedding=_embeddings,
        pinecone_api_key=pinecone_api_key
    )

def create_vectorstore(splits, pinecone_api_key, index_name):
    """Initialize Pinecone, ensure index exists, and store document chunks."""
    if not pinecone_api_key or "your_" in pinecone_api_key.lower():
        raise ValueError("Valid PINECONE_API_KEY is required.")
    if not index_name:
        raise ValueError("PINECONE_INDEX name is required.")

    embeddings = get_embeddings()
    pc = get_pinecone_client(pinecone_api_key)
    
    # Check if index exists, if not create it
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        # Create a new index
        # Note: all-MiniLM-L6-v2 has 384 dimensions
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
            
    vectorstore = PineconeVectorStore.from_documents(
        splits, 
        embeddings, 
        index_name=index_name, 
        pinecone_api_key=pinecone_api_key
    )
    return vectorstore
