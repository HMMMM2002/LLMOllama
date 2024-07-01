import chromadb
from .data_utils import DataUtils
from chromadb.api import ClientAPI

class BetekkChromaDB():
    @classmethod
    def get_chromadb_client_api(cls,database_path:str="/home/orin/Documents/chromaDB") ->ClientAPI:
        return chromadb.PersistentClient(path=database_path)

    