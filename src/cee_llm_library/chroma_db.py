import chromadb
from .data_utils import DataUtils

class BetekkChromaDB:
    def dataBase(collectionName):
        client=chromadb.PersistentClient(path="/home/orin/Documents/chromaDB")
        #client2=chromadb.HttpClient(host='localhost',port=8000)
        collection= client.get_or_create_collection(name=collectionName,embedding_function=DataUtils.embeddingModel())
        #collection = client.get_collection(name=collectionName, embedding_function=DataUtils.embeddingModel())
        return collection
    