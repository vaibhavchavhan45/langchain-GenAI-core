from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Write the 5 facts about topic {topic} in detailed manner with proper information \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

prompt = template.invoke({'topic' : 'Titanic shipwreck'})

# print(prompt) # Write the 5 facts about topic Titanic shipwreck in detailed manner with proper information \n Return a JSON object.'

result = model.invoke(prompt) # json format output
# print(result.content)       # json object without metadata

parsed_result = parser.parse(result.content)  # parse **{json}** to **parsed_result = {obj}** (parsing content no metadata)

print(parsed_result)
print(type (parsed_result))