import os
from dotenv import load_dotenv
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# === Paths and Constants ===
CHROMA_PATH = "index/chroma"
EMBED_MODEL_NAME = "all-mpnet-base-v2"

# === Embedding model (HuggingFace local) ===
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.embed_model = embed_model

# === Chroma vector store ===
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection("legal_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# === Create retriever from index ===
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=3)
