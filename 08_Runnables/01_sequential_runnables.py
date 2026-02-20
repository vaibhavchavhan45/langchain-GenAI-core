# Problem statement
# Write the advantages of given topic and explain those advantages, also return that explanation 
# (It will not return benefits of topic, just returns the explanation)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

template1 = PromptTemplate(
    template = 'Write a benefits or advantages or pros of given provided thing for upcoming future \n {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = 'Explain the following benefits of the given provided text \n {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = RunnableSequence(template1, model, parser, template2, model, parser)

result = chain.invoke({'topic' : "AI"})

print(result)

# feat: explain a two-step LLM pipeline for future benefits using sequencial chain