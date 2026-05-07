from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

url = "https://www.bbc.com/sport/football/articles/cj69lj0w858o"
loader = WebBaseLoader(url)

docs = loader.load()

parser = StrOutputParser()

template = PromptTemplate (
    template= 'Answer the question \n {question} \n from the given text \n {text}',
    input_variables=['question', 'text']
)

chain = template | model | parser

result = chain.invoke({'question':'Who hit the most number of goals in Premier League by English players this season?', 'text' : docs[0].page_content})

print(result)