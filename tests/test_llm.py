from cee_llm_library import DataUtils,BetekkChromaDB

from uuid import uuid4

def test_llm():
    modelId="phi3:14b"
    doc="documents/sample_file2.pdf"
   

    print('__________________________')
    
    query='What are the responsibilities of a research engineer?'
    documents=DataUtils.loadDocument(doc)
    chunks=DataUtils.splitDocuments(documents)
    [print("type of chunk:{position} | {type}".format(position=position,type=type(chunk))) for position,chunk in enumerate(chunks)]
    ##content=[chunk.page_content for chunk in chunks]
    embedding=DataUtils.embeddingModel()
    
    TestDB=BetekkChromaDB.database("my_collection_1")
    
    collection=TestDB.add(
        documents=chunks, 
        metadatas=[{"source": "notion"} for i in range(len(chunks))], # filter on these!
        ids=[uuid4() for i in range(len(chunks))] )

    # #dbChroma=VectorStore.loadIntoDB(chunks,embedding)
    # ##context=dbChroma.search(query,dbChroma)   
    # context=VectorStore.search(query,dbChroma)
    
    # template=OllamaLLM.prompt_teplate(context,query)
    # ##retrieverChroma=dbChroma.as_retriever(dbChroma)
    # retrieverChroma=dbChroma.as_retriever()
    # ##qa=RetrievalQA.from_chain_type(model,chain_type="stuff",retriever=retrieverChroma)
    # ##https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html#langchain.chains.retrieval_qa.base.RetrievalQA
    # qa=RetrievalQA.from_chain_type(llm=Ollama(model=modelId), chain_type="stuff",retriever=retrieverChroma)
    # output=qa.invoke(template)
    # answer=output['result']
    # print(answer)

if __name__ == "__main__":
    test_llm()