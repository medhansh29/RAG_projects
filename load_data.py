from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param

client = MongoClient(key_param.MONGO_URI, server_api = ServerApi('1'))
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

try:
    client.admin.command('ping')
    print("Connected to MongoDB")

    loader = DirectoryLoader('./news_articles', glob="./*.txt", show_progress=True)
    data = loader.load()

    embeddings = OpenAIEmbeddings(openai_api_key = key_param.openai_api_key)

    vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection)
finally:
    client.close()
    print("MongoDB connection closed")



