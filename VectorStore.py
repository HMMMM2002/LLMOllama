from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

class VectorStore(): 
    def loadIntoDB(texts,embeddingModel):
        ids = [str(i) for i in range(1, len(texts) + 1)]
        db=Chroma.from_documents(texts,embeddingModel,ids=ids)
        return db
    
    def search(query,db,k=5):
        docs=db.similarity_search(query,k)
        return docs
    
    def searchWithCosineScore(query,db,k=5):
        docs=db.similarity_search_with(query,k)
        return docs
    
    def retriever(db,search_type='similarity',search_kwargs={'k': 4,})
        retriever=db.as_retriever(search_type,search_kwargs)
        return retriever