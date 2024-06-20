from LLMModel import OllamaLLM
def main():
    modelId="phi3:14b"
    input="why is the sky blue?"
    model=OllamaLLM(modelId)
    prompt=OllamaLLM.generate_prompt(input)
    model.chatWithLLM(prompt)

if __name__ == "__main__":
    main()