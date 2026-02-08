from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me name, age, city, residential address of a fictional person \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)