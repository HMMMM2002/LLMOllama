from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

class VectorStore(): 
    def loadIntoDB(texts,embeddingModel):
        ids = [str(i) for i in range(1, len(texts) + 1)]
        db=Chroma.from_documents(texts,embeddingModel,ids=ids)
        return db
    
    def search():
        