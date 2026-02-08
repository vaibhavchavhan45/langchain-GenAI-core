from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1' 
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = '''Break down 5 jaw-dropping statistics about {topic} that prove it's crazier than you think \n {format_instruction}''',
    input_variables = ['topic'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic' : 'Social Media Addiction'})

print(result)