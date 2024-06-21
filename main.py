from LLMModel import OllamaLLM
from DataUtils import DataUtils
from VectorStore import VectorStore
from langchain.chains import RetrievalQA
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama


def main():
    modelId="phi3:14b"
    #embedModelId="nomic-embed-text-v1.5"
    input="why is the sky blue?"
    #doc="/home/orin/CODES/ML/llmLocal/LLMOllama/documents/CR112_Vol 1_ITT.pdf"
    doc="/home/orin/CODES/ML/llmLocal/LLMOllama/documents/01_CR112_ITT_MainText.pdf"
    model=OllamaLLM(modelId)
    prompt=OllamaLLM.generate_prompt(input)
    #model.chatWithLLM(prompt)

    print('__________________________')
    
    query='what does the Introducrion and Eligibility talk about?'
    documents=DataUtils.loadDocument(doc)
    chunks=DataUtils.splitDocuments(documents)
    #content=[chunk.page_content for chunk in chunks]
    embedding=DataUtils.embeddingModel()
    dbChroma=VectorStore.loadIntoDB(chunks,embedding)
    #context=dbChroma.search(query,dbChroma)
    context=VectorStore.search(query,dbChroma)
    template=OllamaLLM.prompt_teplate(context,query)
    #retrieverChroma=dbChroma.as_retriever(dbChroma)
    retrieverChroma=dbChroma.as_retriever()
    #qa=RetrievalQA.from_chain_type(model,chain_type="stuff",retriever=retrieverChroma)
    #https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html#langchain.chains.retrieval_qa.base.RetrievalQA
    qa=RetrievalQA.from_chain_type(llm=Ollama(model=modelId), chain_type="stuff",retriever=retrieverChroma)
    output=qa.invoke(template)
    answer=output['result']
    print(answer)

if __name__ == "__main__":
    main()