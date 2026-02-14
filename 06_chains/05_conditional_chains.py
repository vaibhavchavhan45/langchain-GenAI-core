from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda 
from typing import Literal
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

class Feedback (BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description = 'Sentiment of the feedback')

parser = StrOutputParser()

pydantic_parser = PydanticOutputParser(pydantic_object = Feedback)

template = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}', 
    input_variables = ['feedback'],
    partial_variables = {'format_instruction' : pydantic_parser.get_format_instructions()}
)

classifier_chain = template | model | pydantic_parser

# result = classifier_chain.invoke({'feedback' : 'This cell is worst'})

# print(result)

positive_template = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)

negative_template = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {feedback}',
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', positive_template | model | parser),
    (lambda x: x.sentiment == 'negative', negative_template | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : 'this phone is devastated'})

print(result)



