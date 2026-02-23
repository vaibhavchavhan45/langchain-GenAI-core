            # Problem statement
# Generates a one-line product description using LLM,
# validates it (word limit), retries on failure,
# and always returns the last LLM response.


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

template = PromptTemplate(
    template="Create a one-line product description for {product_name}",
    input_variables=["product_name"]
)

parser = StrOutputParser()

# chain to generate raw LLM response
product_gen_chain = RunnableSequence(template, model, parser)

# validation logic for LLM (used by RunnableLambda)
def validate_output(text):
    word_count = len(text.split())
    return {
        "word_count" : word_count,
        "is_valid": 0 < word_count <= 20
    }

# run response and validation in parallel
parallel_chain = RunnableParallel({
    'response' : RunnablePassthrough(),
    'validation' : RunnableLambda(validate_output)
})

# combine generation and validation
chain = RunnableSequence(product_gen_chain, parallel_chain)

# retry logic : return valid result early, else return last LLM output
def run_with_retries(input_data, max_retry = 2):
    for _ in range(max_retry):
        result = chain.invoke(input_data)
        last_result = result
        if result["validation"]["is_valid"]:
            return result
    return last_result

# invoke chain with input (input to retry function)
result = run_with_retries(
    {"product_name" : 'Smart Fitness Watch with Heart Rate Monitor'}
)

# print(result)
print(f"{result['response']} \n {result['validation']}")

