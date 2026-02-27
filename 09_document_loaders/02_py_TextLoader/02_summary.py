from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

# Loading the text file
with open("01_lang.txt", encoding = "utf-8") as f:
    text = f.read()

# docs = [
#     Document(
#         metadata = {'source' : "01_lang.txt"},
#         page_content = text
#     )
# ]

template = PromptTemplate(
    template = 'Generate the summary oog th following text \n {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = RunnableSequence(template, model, parser)

result = chain.invoke(text)

print(result)