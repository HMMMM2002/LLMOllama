from cee_llm_library import BetekkChromaDB, DataUtils,OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
# from langchain.chains import VectorDBQA
# from langchain.vectorstores import Chroma
from uuid import uuid4
from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from chromadb.utils import embedding_functions
import json

DEFAULT_LLM_MODEL_ID="phi3:14b"
DEFAULT_DOC_NAME="documents/sample_file2.pdf"
DEFAULT_COLLECTION_NAME = "collection-1"
DEFAULT_CHROMADB_DATA_DIRECTORY="/home/orin/Documents/chromaDB"
DEFAULT_EMBEDDING_MODEL_NAME = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

def test_llm():
    
   

    print('__________________________')
    
    query='What are the responsibilities of a research engineer?'
    documents=DataUtils.load_document(DEFAULT_DOC_NAME)
    assert isinstance(documents, list)
    for document in documents:
        assert isinstance(document, Document)
    chunks=DataUtils.split_documents(documents)
    [print("type of chunk:{position} | {type}".format(position=position,type=type(chunk))) for position,chunk in enumerate(chunks)]
    ##content=[chunk.page_content for chunk in chunks]
    embedding_model=DataUtils.get_embedding_model()
    assert isinstance(embedding_model,Embeddings)

    test_client=BetekkChromaDB.get_chromadb_client_api()
    embedding_function=embedding_functions.OllamaEmbeddingFunction(url=DEFAULT_OLLAMA_URL,model_name=DEFAULT_EMBEDDING_MODEL_NAME)
    collection: Collection = test_client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME,
                                                                  metadata={"collection_name": DEFAULT_COLLECTION_NAME},
                                                                  embedding_function=embedding_function)
    assert isinstance(collection,Collection)

    list_of_metadata: list[dict] = []
    list_of_id: list[str] = []
    chunks_json: list = []
    for i, chunk in enumerate(chunks):
        try:
            chunk_json = json.dumps(chunk.page_content)
            chunks_json.append(chunk_json)
            list_of_metadata.append({"source": DEFAULT_DOC_NAME})
            list_of_id.append(str(uuid4()))
        except Exception as e:
            print(f"Error converting chunk to JSON: {e}")
            
            continue

    # [print(chunk.page_content,type(chunk.page_content)) for chunk in chunks if not isinstance(chunk.page_content,str)

    collection.add(
        documents=[documents,], 
        metadatas=list_of_metadata, # filter on these!
        ids=list_of_id
    )

    context=collection.query(
    query_texts=[query],
    n_results=5,
    where={"metadata_field": DEFAULT_DOC_NAME},
    where_document={"$contains":"search_string"}
    )


    # #dbChroma=VectorStore.loadIntoDB(chunks,embedding)
    # ##context=dbChroma.search(query,dbChroma)   
    # context=VectorStore.search(query,dbChroma)
    
    template=OllamaLLM.prompt_teplate(context,query)
    # ##retrieverChroma=dbChroma.as_retriever(dbChroma)
    retrieverChroma=test_client.as_retriever()
    # ##qa=RetrievalQA.from_chain_type(model,chain_type="stuff",retriever=retrieverChroma)
    # ##https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html#langchain.chains.retrieval_qa.base.RetrievalQA
    qa=RetrievalQA.from_chain_type(llm=Ollama(model=DEFAULT_LLM_MODEL_ID), chain_type="stuff",retriever=retrieverChroma)
    output=qa.invoke(template)
    answer=output['result']
    print(answer)

if __name__ == "__main__":
    test_llm()