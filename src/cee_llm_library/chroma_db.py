import chromadb
from .data_utils import DataUtils

class BetekkChromaDB:
    @classmethod
    def database(cls,collection_name:str):
        return chromadb.PersistentClient(path="/home/orin/Documents/chromaDB")

    