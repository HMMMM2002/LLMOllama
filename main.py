from LLMModel import OllamaLLM
from DataUtils import DataUtils
def main():
    modelId="phi3:14b"
    embedModelId="nomic-embed-text-v1.5"
    input="why is the sky blue?"
    model=OllamaLLM(modelId)
    prompt=OllamaLLM.generate_prompt(input)
    model.chatWithLLM(prompt)

if __name__ == "__main__":
    main()