from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

class News(BaseModel):
    news_topic: str = Field(description = 'Extract the topic from provided article or text')
    date: str = Field(description = 'The mentioned date in text when that particular senario happened in format DD/MM/YY')
    summary: str = Field(description = 'Generate summary in 2 to 3 points') 
    headlines: str = Field(description = 'Generate catchy headlines in context of input article')
    sub_headline: str = Field(description = 'Generate subheadline followed by headlines')

parser = JsonOutputParser(pydantic_object = News)

template = PromptTemplate(
    template = 'Analyze the following news article: {article} and generate news_topic, date, summary, headlines, sub_headlines in provided schema \n {format_instruction}',
    input_variables = ['article'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

text = '''
Tesla announced a major breakthrough in battery technology today, revealing a new solid-state battery that could potentially double the range of electric vehicles while cutting charging time in half. The company's CEO stated during the unveiling event in Austin, Texas that the new batteries will enter mass production by late 2025. Industry experts are calling this a game-changer for the EV market, though some remain skeptical about the ambitious timeline. The announcement caused Tesla's stock to surge 8% in after-hours trading, with analysts predicting this could accelerate the automotive industry's transition away from fossil fuels. However, competitors like Toyota and BMW have also been working on similar technology, setting up what could be an intense race to bring solid-state batteries to market first.'''

chain = template | model | parser

result = chain.invoke(text)

print(result)

# Visualisation Representation of chain
chain.get_graph().print_ascii()