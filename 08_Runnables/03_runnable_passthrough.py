# Problem statement :
# Build a joke generation where llm response should be joke and it's description

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

template = PromptTemplate(
    template = 'Write a joke on the provided topic {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = 'Write the explanation for the given provided joke {joke}',
    input_variables = ['joke']
)
parser = StrOutputParser()

joke_generation_chain = RunnableSequence(template, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'Explanation' : RunnableSequence(template2, model, parser)
})

final_chain = RunnableSequence(joke_generation_chain, parallel_chain)

result = final_chain.invoke({'topic' : 'Pakistan'})

print(result)
print("JOKE : ", result['joke'])
print("Explanation : ", result['Explanation'])

