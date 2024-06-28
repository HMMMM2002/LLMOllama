#from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.llms import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
import os

class DataUtils:
    def loadDocument(inputDocument):
        file_path= inputDocument
        file_path_and_extension: tuple = os.path.splitext(file_path)
        file_extension = file_path_and_extension[1]
        loader= None
        print("file_extension: found ", file_path_and_extension)
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(inputDocument)
        elif file_path_and_extension == '.csv':
            loader = CSVLoader(inputDocument)
        elif file_extension == '.xlsx':
            loader = UnstructuredExcelLoader(inputDocument)
      
        if loader is None:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        content = loader.load()
        return content
    
    #https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#langchain-text-splitters-character-recursivecharactertextsplitter
    #https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
    def splitDocuments(inputcontent,chunk_size=200,chunk_overlap=50,):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,)
        #splitter = RecursiveCharacterTextSplitter()
        docs=splitter.split_documents(inputcontent)
        return docs


    def embeddingModel(model="nomic-embed-text"):
        embedding_function = OllamaEmbeddings(model=model)
        #NomicEmbeddings(model=model,dimensionality=dimensionality)
        #https://ollama.com/library/nomic-embed-text
        #https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.ollama.OllamaEmbeddings.html
        return embedding_function
    

        
