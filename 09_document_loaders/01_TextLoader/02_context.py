# TextLoader --> deprecated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

loader = TextLoader('cric.txt', encoding = 'utf-8')
docs = loader.load()

parser = StrOutputParser()

template = PromptTemplate(
    template = 'Write the context of the given text - \n {text}',
    input_variables = ['text']
)

chain = template | model | parser

result = chain.invoke(docs[0].page_content)

print(result)

