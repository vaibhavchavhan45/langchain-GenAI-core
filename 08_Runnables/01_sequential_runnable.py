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
    template = 'Write a joke about the given topic {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = 'Explain the following joke in brief {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = RunnableSequence(template1, model, parser, template2, model, parser)

result = chain.invoke({'topic' : "AI will replace the Software Engineer"})

print(result)

# feat: generate a joke description using sequential runnable from a topic