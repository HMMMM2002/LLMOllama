import ollama
class OllamaLLM:
    """
    A class to interface with the Ollama language model for generating and handling conversational prompts.

    Attributes:
        modelId (str): The identifier for the specific model version used for generating responses. Default is 'phi3:14b'.

    Methods:
        generate_prompt(input):
            Generates a prompt formatted for the Ollama language model.

            Args:
                input (str): The input string from the user.

            Returns:
                list: A list of dictionaries representing the formatted prompt.
        
        chatWithLLM(self, prompt):
            Communicates with the Ollama language model using the provided prompt and prints the response.

            Args:
                prompt (list): A list of dictionaries formatted as a prompt for the Ollama language model.
    """


    def __init__(self,modelId='phi3:14b'):
        self.modelId=modelId

    def generate_prompt(input):
        messages=[
            {"role":"system","content":"Your are a developer."},
            {"role":"user","content":input},
        ]
        return messages
        
    def chatWithLLM(self,prompt):
        response = ollama.chat(model=self.modelId,messages=prompt)
        print(response['message']['content'])