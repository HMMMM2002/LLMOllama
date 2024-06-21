#from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
#from langchain.vectorstores import MongoDBAtlasVectorSearch
#from langchain.document_loaders import DirectoryLoader
from langchain.llms import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

class DataUtils:
    def loadDocument(inputDocument):
        loader = PyPDFLoader(inputDocument)
        content = loader.load()
        return content
    
    #https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#langchain-text-splitters-character-recursivecharactertextsplitter
    #https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
    def splitDocuments(inputcontent,chunk_size=200,chunk_overlap=50,length_funtion=len,is_separator_regex=False,):
        #splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,length_funtion=length_funtion,is_separator_regex=False,)
        splitter = RecursiveCharacterTextSplitter()
        docs=splitter.split_documents(inputcontent)
        return docs
    
    def embeddingModel(model="nomic-embed-text"):
        embedding_function = OllamaEmbeddings(model=model)
        #NomicEmbeddings(model=model,dimensionality=dimensionality)
        #https://ollama.com/library/nomic-embed-text
        #https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.ollama.OllamaEmbeddings.html
        return embedding_function
    

        
