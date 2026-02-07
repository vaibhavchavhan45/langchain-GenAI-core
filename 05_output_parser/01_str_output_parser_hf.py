# Prompt -- LLM -- detailed report -- Prompt(detailed report) -- LLM -- Output (summary)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation',
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

prompt = template.invoke({'topic' : 'milky way'})

initial_result = model.invoke(prompt)

final_template = PromptTemplate(
    template = 'Write a 5 to 10 lines summary on the {report}',
    input_variables = ['report']
)

final_prompt = final_template.invoke({'report' : initial_result})

final_result = model.invoke(final_prompt)

print(initial_result.content)
print('='*35)
print(final_result.content)

