import os
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Download NLTK data only if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize your embedding model once (choose model you want)
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize Chroma client with persistence (adjust path as needed)
chroma_client = chromadb.PersistentClient(path="index/chroma")

# Create or get collection to store vectors and metadata
collection = chroma_client.get_or_create_collection(name="legal_docs")

def chunk_text_hybrid(text: str, max_tokens: int = 500, overlap: int = 100) -> List[str]:
    """
    Chunk text by sentences with sliding window token limit + overlap.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        token_len = len(sentence.split())  # crude token count, can be replaced by tokenizer tokens
        if current_len + token_len > max_tokens:
            chunks.append(' '.join(current_chunk))
            # slide window back by overlap tokens (sentences)
            overlap_sentences = []
            overlap_len = 0
            while current_chunk and overlap_len < overlap:
                last_sentence = current_chunk.pop()
                overlap_sentences.insert(0, last_sentence)
                overlap_len += len(last_sentence.split())
            current_chunk = overlap_sentences + [sentence]
            current_len = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_len += token_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def embed_and_store(docs: List[Tuple[str, str]]):
    print(f"Processing {len(docs)} documents for embedding and storage...")
    batch_size = 96
    all_chunks = []
    all_metadatas = []
    all_ids = []

    doc_counter = 0
    for source_file, text in docs:
        doc_counter += 1
        chunks = chunk_text_hybrid(text)
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue
            clean_name = source_file.replace('/', '_').replace('\\', '_')
            chunk_id = f"{clean_name}_chunk_{i}_{len(all_ids)}"

            all_chunks.append(chunk)
            all_metadatas.append({"source_file": source_file, "chunk_id": chunk_id})
            all_ids.append(chunk_id)

        if doc_counter % 100 == 0:
            print(f"Chunked and prepared {doc_counter} documents...")

    print(f"Total chunks to embed: {len(all_chunks)}")
    num_batches = (len(all_chunks) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(all_chunks), batch_size), total=num_batches, desc="Embedding Batches"):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_embeddings = embedder.encode(batch_chunks, show_progress_bar=False, convert_to_numpy=True)
        batch_ids = all_ids[i:i+batch_size]
        batch_metadatas = all_metadatas[i:i+batch_size]

        collection.upsert(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    print("Embedding and storage completed.")
   

# Example usage (you can call this after loading your data in load_all_data.py):
if __name__ == "__main__":
    from load_all_data import load_json_files, load_csv_files, load_pdf_files

    json_docs = load_json_files("data/jsons")
    csv_docs = load_csv_files("data/csvs")
    pdf_docs = load_pdf_files("data/pdfs")

    all_docs = json_docs + csv_docs + pdf_docs

    embed_and_store(all_docs)
