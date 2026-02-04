from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key =  os.getenv("GROQ_API_KEY"),
    openai_api_base = "https://api.groq.com/openai/v1"
)

## Schema
# class Review(TypedDict):
#     summary: str
#     sentiment: str

# OR

class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Classify the sentiment of the review as < Positive | negative | neutral >"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """ I order an phone of your brand, the hardware is great but, software feels bloated with too many pre-installed app which I can't remove and the UI looks outdated as compared to other brands, Hope the software update will solve the issue """
)

print(result)
print(result['summary'])
print(result['sentiment'])

