import random
class DummyLLM:
    def __init__(self):
        print('LLM created')

    def predict(self, prompt):
        response_list = [
            'Delhi is capital of India',
            'ICC stands for International Cricker Council',
            'AI can be used on vast scale in day to day life'
        ]

        return {'response' : random.choice(response_list)}
    
llm = DummyLLM()
llm.predict('Capital of India')