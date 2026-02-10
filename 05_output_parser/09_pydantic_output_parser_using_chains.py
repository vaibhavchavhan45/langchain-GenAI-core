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
    name: str = Field(description = 'name of the monument')
    location: str = Field(description = 'location of the monument including city, country, co-ordinates')
    year_built: int = Field(description = 'The year in which monument was built')
    architectural_style: str = Field(description = 'architectural style of the monument like Gothic, Baroque, Modern etc.')
    purpose: str = Field(description = 'Backstory behind the creation of monuments like religious, memorial, royal residence etc.')
    dimensions: str = Field(description = 'Height, area, notable measurements of the monuments')
    materials_used : str = Field(description = 'materials was used while building the monument like Stone, marble, steel, etc.')
    historical_significance: str = Field(description = 'As a historial purpose why the monument is still important')
    cultural_impact: str = Field(description = 'How monument influence society, art, diversity of culture')
    interesting_facts: str = Field(description = 'interesting facts related to monument like hidden, secret, mysterious records')
    visitors: int = Field(description='Average number of visitors per year')

parser = PydanticOutputParser(pydantic_object = Character)

template = PromptTemplate(
    template = '''Provide comprehensive information about {monument} including its official name, exact location, year of construction, architectural style, original purpose, key dimensions, materials used, historical significance, cultural impact, and three fascinating lesser-known facts \n {format_instructions}''',
    input_variables = ['monument'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'monument' : 'Taj Mahal'})

print(result)