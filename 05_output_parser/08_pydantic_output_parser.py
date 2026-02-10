from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parser import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

class Character(BaseModel):
    character_name: str = Field(description = 'name of the fictional character')
    age: int = Field(description = 'present age of character')
    physical_description: str = Field(description = 'apperance, height, feature of the character')
    personality_traits: str = Field(description = 'key characteristics of the character')
    key_quotes: str = Field(description = 'Famous and impactful lines said of the character')
    motivations: str = Field(description = 'What drive the character for this role')
    strength: str = Field(description = 'super power that the character has')
    symbolic_significance: str = Field(description = 'the symbol which character represent')
    character_arc: str = Field(description = 'How the character changes into their own arc')
    
parser = PydanticOutputParser(pydantic_object = Character)

template = PromptTemplate(
    template = '''Provide detailed analysis of a major character from {novel} including their name, age, physical description, personality traits, key quotes, strength, character development, and symbolic significance \n {format_instruction}''',
    input_variables = ['novel'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

prompt = template.invoke({'novel': 'The Great Gatsby'})

result = model.invoke(prompt)

parsed_result = parser.parse(result.content)

print(parsed_result)