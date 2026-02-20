from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

passthrough = RunnablePassthrough()

# e.g. 1
input = passthrough.invoke(2)
print(input)

# e.g. 2
input2 = passthrough.invoke({'name': "Vaibhav"})
print(input2)

# e.g. 3
template = PromptTemplate(
    template = 'Write a well detailed structured report on the topic {topic}',
    input_variables = ['topic']
)

chain = template | passthrough

result = chain.invoke({'topic' : "Robotics"})

print(result)


# RunnablePassthrough returns whatever it receives as input