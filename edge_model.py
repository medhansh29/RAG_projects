import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json

# ---- Config ----
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CORPUS_FILE = "corpus.txt"
TOP_K = 5
OLLAMA_MODEL = "mistral"


# ---- Step 1: Load and Chunk Corpus ----
def load_corpus(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return [text[i:i+300] for i in range(0, len(text), 250)]


# ---- Step 2: Create Embeddings ----
def embed_chunks(chunks, model):
    return model.encode(chunks, show_progress_bar=True)


# ---- Step 3: Build FAISS Index ----
def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


# ---- Step 4: Retrieve Relevant Chunks ----
def retrieve_chunks(query, embedder, index, chunks, top_k):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]


# ---- Step 5: Generate Answer via Ollama ----
def generate_answer(context, question):
    prompt = f"""You are a helpful assistant.

Context:
{context}

Question: {question}
Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
        stream=True
    )

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                part = json.loads(line.decode("utf-8"))
                full_response += part.get("response", "")
            except json.JSONDecodeError:
                continue
    return full_response.strip()


# ---- Main Pipeline ----
def main():
    start_time = time.time()

    print("🔹 Loading corpus...")
    chunks = load_corpus(CORPUS_FILE)

    print("🔹 Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("🔹 Embedding chunks...")
    chunk_embeddings = embed_chunks(chunks, embedder)

    print("🔹 Building FAISS index...")
    index = build_faiss_index(np.array(chunk_embeddings))

    setup_time = time.time() - start_time
    print(f"⏱️ Setup completed in {setup_time:.2f} seconds.")

    while True:
        print("\n🔍 Ask a question (or type 'exit'):")
        query = input(">> ").strip()
        if query.lower() == "exit":
            break

        query_start = time.time()

        print("🔹 Retrieving relevant chunks...")
        retrieved = retrieve_chunks(query, embedder, index, chunks, TOP_K)
        context = "\n\n".join(retrieved)

        print("🧠 Generating answer from Ollama...\n")
        answer = generate_answer(context, query)

        query_time = time.time() - query_start
        print(f"🗣️ Answer:\n{answer}")
        print(f"⏱️ Query processed in {query_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
