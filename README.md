📘 Question Answering and Movie Recommendation System using MongoDB Atlas & Vector Search
This repository contains three Python scripts demonstrating how to build:

A RAG-based Question Answering App using OpenAI and MongoDB Atlas Vector Search

A Movie Recommendation System using Sentence Transformers and MongoDB’s native vector search

🗂 Files Overview
1. load_data.py — Data Loader
This script loads .txt files from the news_articles directory, converts them to embeddings using OpenAI, and stores them in a MongoDB Atlas collection using vector indexing.

Key Features:
Loads documents using LangChain's DirectoryLoader

Embeds documents using OpenAIEmbeddings

Stores them in MongoDB Atlas Vector Search format

How to Run:
bash
Copy
Edit
python load_data.py
Ensure news_articles/ contains .txt files before running.

2. extract_information.py — Vector-based QA Application
This Gradio web app allows users to ask questions and get answers grounded only in the uploaded text documents (no external knowledge). It uses:

LangChain's RetrievalQA with a custom prompt

MongoDB Atlas Vector Search to retrieve context

OpenAI LLM to generate responses strictly based on retrieved content

Key Features:
RAG (Retrieval Augmented Generation) pattern

Execution time logging

Gradio UI interface

How to Run:
bash
Copy
Edit
python extract_information.py
UI Preview:
Textbox: User inputs question

Output 1: Answer from vector retrieval + LLM (grounded)

Output 2: Execution time

⚠️ Requires that load_data.py has been run to load the data.

3. movie_recs.py — Movie Plot Semantic Search
A standalone script that performs movie recommendation based on semantic similarity of movie plots using sentence-transformers and MongoDB’s $vectorSearch.

Key Features:
Embeds movie plots from MongoDB’s sample_mflix.movies

Uses all-MiniLM-L6-v2 model for embeddings

Searches for similar plots using native vector search

How to Run:
bash
Copy
Edit
python movie_recs.py
⚠️ Assumes your MongoDB collection has precomputed plot embeddings under plot_embedding_hf.

🔧 Setup & Configuration
Create a key_param.py file with:

python
Copy
Edit
openai_api_key = "YOUR_OPENAI_KEY"
MONGO_URI = "YOUR_MONGODB_ATLAS_URI"
Install dependencies:

bash
Copy
Edit
pip install pymongo langchain openai gradio sentence-transformers
