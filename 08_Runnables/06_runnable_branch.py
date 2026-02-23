# Problem Statement :
# Provide a topic to LLM and ask to generate detailed report, if that report is greater than 500 words then summarise it, else print that response as an output 

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

template = PromptTemplate(
    template = 'Write the detailed report on the topic {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = 'Summarize the given provided text \n {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

report_gen_chain = template | model | parser

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 500, template2 | model | parser),
    RunnablePassthrough()
)

chain = RunnableSequence(report_gen_chain, branch_chain)

result = chain.invoke({'topic' : "GenZ protest in Nepal"})

print(result)


# Note :: Sequential chain can be created like :
# 1.  Sequential(template, model, parser)
# OR
# 2. template | model | parser