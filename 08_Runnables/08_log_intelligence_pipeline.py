# Problem statement :
# Raw system errors are hard to understand and do not clearly indicate what action is required.
# This system reads a system error, understands what went wrong, determines how serious the issue is, 
# and then suggests the appropriate next steps (incident response or troubleshooting) based on severity.

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser, 
    PydanticOutputParser
)
from pydantic import (
    Field, 
    BaseModel
)
from langchain_core.runnables import (
    RunnableSequence, 
    RunnableLambda, 
    RunnableBranch
)

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1',
    temperature = 0.5
)

# generate summary and define degree/level of the problem
class InterpretedLog(BaseModel):
    summary: str = Field(..., description = "Explanation of error in simple English")
    severity: str = Field(
        ...,
        decription = "severity level",
        regex = "LOW | MEDIUM | HIGH | CRITICAL"
    )

pydantic_parser = PydanticOutputParser(pydantic_object = InterpretedLog)

str_parser = StrOutputParser()

# Prompt: Interpret log + classify severity
error_prompt = PromptTemplate(
    template = (
        "Analyze the system log below.\n"
        "1. Explain what happened in simple English\n"
        "2. Classify severity as one of: LOW, MEDIUM, HIGH, CRITICAL\n\n"
        "{format_instruction} \n"
        "System logs : \n {log}"
    ),
    input_variables = ['log'],
    partial_variables = {'format_instruction' : pydantic_parser.get_format_instructions()}
)

# Chain that converts raw log -> structured meaning
error_chain = RunnableSequence(error_prompt, model, pydantic_parser)

# Prompt: For serious issues/incidents
high_severity_prompt = PromptTemplate(
    template = (
        "The following issue is CRITICAL.\n"
        "Provide clear incident response steps for an on-call engineer.\n\n"
        "Summary : \n {summary}"
    ), 
    input_variables = ['summary']
)

# Chain to generate high response steps
high_severity_chain = RunnableSequence(high_severity_prompt, model, str_parser)

# Prompt: For non-serious issues/just troubleshooting
low_severity_prompt = PromptTemplate(
    template = (
        "The following issue is NOT CRITICAL \n"
        "Provide basic troubleshooting steps \n\n"
        "Summary : \n {summary}"
    ),
    input_variables = ['summary']
)

# chain for troubleshoot
low_severity_chain = RunnableSequence(low_severity_prompt, model, str_parser)

# severity check
def is_critical(log: InterpretedLog) -> bool:
    return log.severity in {"HIGH, CRITICAL"}

severity_check = RunnableLambda(is_critical)

# if severith_check is HIGH | CRITICAL then go to high severity chain otherwise go to low severity chain
branch_chain = RunnableBranch(
    (severity_check, high_severity_chain),
    low_severity_chain
)

# First chain understands and classify the log (summary + severity)., 
# branch then decides what to do next based on that severity
chain = RunnableSequence(error_chain, branch_chain)

result = chain.invoke({
    "log": "ERROR 504: Gateway Timeout while calling user-service from API gateway"
})

print(result)