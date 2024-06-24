from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

class VectorStore(): 
    def loadIntoDB(texts,embeddingModel):
        ids = [str(i) for i in range(1, len(texts) + 1)]
        db=Chroma.from_documents(texts,embedding=embeddingModel,ids=ids)
        return db
    
    def search(query,db,k=5,search_type="similarity"):
        docs=db.search(query=query,k=k,search_type=search_type)
        return docs
    
    #def searchSimilarity(query,db,k=5):
        #docs=db.search(query=query,search_type="similarity")
        #return docs
    
    #https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.search
    #def retriever(db,search_type='similarity',search_kwargs={'k': 4,}):
        #retriever=db.as_retriever(search_type,search_kwargs)
        #return retriever