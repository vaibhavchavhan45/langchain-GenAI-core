from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(
    model = 'text-embedding-3-large',
    dimensions = 32
)

result = model.embed_query("What is capital of Italy?")

print(str(result))