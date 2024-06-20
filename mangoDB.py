from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import key_param

class mangoDB():
    def mgDB():
        client = MongoClient(key_param.MONGO_URI)
        dbName = "langchain_demo"
        collectionName = "collection_of_text_blobs"
        collection = client[dbName][collectionName]
    
    def VectorStore(data,embdedingModel,collection):
        vectorStore = MongoDBAtlasVectorSearch.from_documents( data, embeddings, collection=collection )
        return vectorStore
    
    def search():
        