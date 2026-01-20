from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model='openai/gpt-oss-120b',
    temperature=0.8,
    max_completion_tokens=100,
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)