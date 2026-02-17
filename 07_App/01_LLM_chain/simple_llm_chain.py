from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(
    model_name = "gpt-3.5-turbo",
    temperature = 0.7 
)

template = PromptTemplate(
    template = 'Suggest a catchy blog title about the topic {topic}',
    input_variables = ['topic']
)

topic = input('Enter a blog topic >> ')

chain = LLMChain(llm=llm, prompt=template)

result = chain.invoke({"topic" : topic})
# OR
# result = chain.run(topic=topic) # .run() deprecated but works

print("Blog title >> ", result)