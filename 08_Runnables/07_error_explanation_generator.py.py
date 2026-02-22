# Problem Statement
# Build a system that takes raw system error logs and automatically converts them into short, easy-to-understand English explanations, 
# fixing the output if it exceeds the allowed length. (allowed length = 60 words)
# if response is greater than 60 words then fix that response in less than 60 words using template2


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableBranch, RunnablePassthrough
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1',
    temperature = 0.7
)

# Enforces structured output and predictable fields
class LogExplanation(BaseModel):
    explanation : str = Field(
        ... ,
        description = "Simple explanation of the system error",
        max_length = 500
    )

# Parser to convert LLM output into ErrorExplanation schema
parser = PydanticOutputParser(pydantic_object = LogExplanation)

# Prompt to generate initial explanation
error_template = PromptTemplate(
    template = (
        "Explain the following error in simple english \n"
        "Rules : \n"
        "1. one paragraph \n" 
        "2. max 60 words \n"
        "{format_instruction} \n\n"
        "Error log : \n {log}"
    ),
    input_variables = ['log'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

# Chain that generates the initial error explanation
error_explain_chain = RunnableSequence(error_template, model, parser) # holds simple explanation of the error

# Prompt to FIX explanation if invalid
explanation_template = PromptTemplate(
    template = (
        'Rewrite the explanation below of the given provided text \n {text}'
        'Rules : \n'
        '1. Simple english \n'
        '2. one paragraph \n'
        '3. max 60 words \n'
        '{format_instruction} \n \n'
    ),
    input_variables = ['text'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

# Chain that fixes an invalid explanation
fix_explain_chain = RunnableSequence(
    RunnableLambda(lambda x: {'text': x.explanation}),
    explanation_template,
    model,
    parser
) # holds the explanation

# validating the explanation
def validate(input_text):
    return len(input_text.explanation.split()) > 60

validate_runnable = RunnableLambda(validate)

# if explanation exceeds more than 60 words then again generate description otherwise print that explanation
branch = RunnableBranch(
    (validate_runnable, fix_explain_chain),
    (RunnablePassthrough())
)

# combining the error explain chain and description explain chain 
chain = RunnableSequence(error_explain_chain, branch)

# final result
result = chain.invoke({'log': "ERROR 504: Gateway Timeout while calling user-service from API gateway"})

print(result)

# commit --> POC for basic system error explanation generator using LLM