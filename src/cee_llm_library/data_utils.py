#from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.llms import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import os

class DataUtils:
    @classmethod
    def load_document(cls,input_file_path:str) -> list[Document]:
        file_path= input_file_path
        file_path_and_extension: tuple = os.path.splitext(file_path)
        file_extension = file_path_and_extension[1]
        loader= None
        print("file_extension: found ", file_path_and_extension)
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(input_file_path)
        elif file_path_and_extension == '.csv':
            loader = CSVLoader(input_file_path)
        elif file_extension == '.xlsx':
            loader = UnstructuredExcelLoader(input_file_path)
      
        if loader is None:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        return loader.load()
    
    #https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#langchain-text-splitters-character-recursivecharactertextsplitter
    #https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
    @classmethod
    def split_documents(cls,input_contents: list[Document],chunk_size: int=200,chunk_overlap:int=50):
        if not isinstance(input_contents, list):
            raise TypeError("Input must be a list of Documents")
        
        for input_content in input_contents:
                if not isinstance(input_content, Document):
                    raise TypeError("Input must be a list of Documents")
                
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,)
        #splitter = RecursiveCharacterTextSplitter()
        docs=splitter.split_documents(input_contents)
        return docs

    @classmethod
    def get_embedding_model(cls,model_name:str="nomic-embed-text") -> Embeddings:
        try:
            embedding_model = OllamaEmbeddings(model=model_name)
            #NomicEmbeddings(model=model,dimensionality=dimensionality)
            #https://ollama.com/library/nomic-embed-text
            #https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.ollama.OllamaEmbeddings.html
            if not isinstance(embedding_model, Embeddings):
                raise TypeError("Embedding model is not an instance of Embeddings")
            return embedding_model
        except Exception as e:
            print(f"Error creating embedding model: {e}")
            raise
        
    

        
