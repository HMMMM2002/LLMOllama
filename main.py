from LLMModel import OllamaLLM
from DataUtils import DataUtils
from mangoDB import mangoDB
from langchain.chains import RetrievalQA

def main():
    modelId="phi3:14b"
    embedModelId="nomic-embed-text-v1.5"
    input="why is the sky blue?"
    model=OllamaLLM(modelId)
    prompt=OllamaLLM.generate_prompt(input)
    model.chatWithLLM(prompt)
    
    
    documents=DataUtils.loadDocument()
    content=DataUtils.splitText(documents)
    embedding=DataUtils.embeddingModel()
    db=mangoDB.VectorStore(content,embedding,mangoDB.mgDB())
    retriever=db.as_retriever()
    qa=RetrievalQA.from_chain_type(model,chain_type="assistant",retriever=retriever)
    result=qa.run(input)

if __name__ == "__main__":
    main()