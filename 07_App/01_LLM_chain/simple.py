from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

template = PromptTemplate(
    template="Suggest catchy blog titles for the topic {topic}",
    input_variables=["topic"]
)

topic = input("Enter the input >> ")

prompt = template.format(topic=topic)
# OR
# prompt = template.invoke({'topic' : topic})

blog_title = model.invoke(prompt)

print("Generated blog title :", blog_title)
