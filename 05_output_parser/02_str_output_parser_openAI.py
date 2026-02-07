from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

# Prompt -- LLM -- detailed report -- Prompt(detailed report) -- LLM -- Output (summary)

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

initial_template = PromptTemplate(
    template = 'Write a full fledge indepth report with impactful explaination, defining simplicity of language on the topic {topic}',
    input_template = ['topic']
)

initial_prompt = initial_template.invoke({'topic' : 'Mystery of Bermuda triangle'})

initial_result = model.invoke(initial_prompt)

final_template = PromptTemplate(
    template = 'Write a summary in 5 to 10 lines on the topic {topic}',
    input_variable = ['topic']
)

final_prompt = final_template.invoke({'topic' : initial_result})

result = model.invoke(final_prompt)

print(initial_result.content)
print('========================================')
print(result.content)


