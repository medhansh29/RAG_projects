import time
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# --- MongoDB Setup ---
# Replace with your actual MongoDB connection URI
uri = "mongodb+srv://medhanshgarg29:Vq51CUEksAs4MnIn@cluster0.y7riqh3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

# --- Sentence Transformer Setup ---
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# --- Connect to DB ---
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB!")

    db = client["sample_mflix"]
    collection = db["movies"]

    # --- Function to generate embedding ---
    def generate_embedding(text: str) -> list[float]:
        return model.encode(text).tolist()

    #for doc in collection.find({'plot':{"$exists" : True}}).limit(50):
        #doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
        #collection.replace_one({'_id':doc['_id']},doc)

    query = "a drama about the courtroom"

    start_time = time.time()

    results = collection.aggregate([
        {"$vectorSearch":{
            "queryVector": generate_embedding(query),
            "path": "plot_embedding_hf",
            "numCandidates" : 100,
            "limit" : 4,
            "index": "PlotSemanticSearch"
        }}
    ])
    

    for document in results:
        print(f'Movie Name: {document["title"]},\nMovie Plot: {document["fullplot"]}\n')

    elapsed_time = time.time() - start_time

    print(f"⏲️ Total Time: {elapsed_time:.4f} seconds")


finally:
    client.close()
    print("🔌 MongoDB connection closed.")
