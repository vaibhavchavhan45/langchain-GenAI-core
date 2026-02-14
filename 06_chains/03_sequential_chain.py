from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import random

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1' 
)

parser = StrOutputParser()

template1 = PromptTemplate(
    template = 'Generate a creative startup name for a {industry} business. Return ONLY the name.',
    input_variables = ['industry']
)

template2 = PromptTemplate(
    template = 'Write a 2-line elevator pitch for this startup name:\n{name}',
    input_variables = ['name']
)

chain = (
        template1 
        | model 
        | parser
        | template2
        | model
        | parser
)

topics = [
    "healthcare AI",
    "fintech",
    "edtech",
    "cybersecurity",
    "agritech",
    "ecommerce",
    "climate tech",
    "gaming",
    "legal tech",
    "logistics"
]

topic = random.choice(topics)
result = chain.invoke({'industry': topic})

print(result)