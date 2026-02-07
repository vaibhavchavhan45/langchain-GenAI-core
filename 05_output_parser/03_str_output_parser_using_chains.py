from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Prompt -- LLM -- detailed report by parser -- Prompt(detailed report by template2) -- LLM -- Output (summary) by parser

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv("GROQ_API_KEY"),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

template1 = PromptTemplate(
    template = 'Write a full fledged detailed explaination report of the topic {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = 'Write the summary of provided topic {text} in 5 to 10 lines \n',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'Theorey of Relativity'})

print(result)