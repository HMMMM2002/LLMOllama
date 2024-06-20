from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataUtils:
    def loadDocument(inputDocument):
        loader = PyPDFLoader(inputDocument)
        content = loader.load()
        return content
    
    def splitText(inputcontent,chunk_size=200,chunk_overlap=50,length_funtion=len,is_separator_regex=False):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size,chunk_overlap,length_funtion,is_separator_regex)
        texts=text_splitter.create_documents(inputcontent)
        return texts

    def embeddingModel(model="nomic-embed-text-v1.5",dimensionality=256):
        embedding_function = NomicEmbeddings(model,dimensionality,)
        return embedding_function
    

        
