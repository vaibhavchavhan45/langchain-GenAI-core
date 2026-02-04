from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv("GROQ_API_KEY"),
    openai_api_base = "https://api.groq.com/openai/v1"
)

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the important key themes from the review"]
    summary: Annotated[str, "The summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Classify the review as < positive | negative > based on the user prompt"]
    pros: Annotated[Optional[list[str]], "Write down all the pros from review provided"]
    cons: Annotated[Optional[list[str]], "Write down all the cons from the review provided"]
    name: Annotated[Optional[str], "Mention the name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""" I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 50x actually works well for distant objects, but anything beyond 15x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by John Doe
""")

print(result)

# Extracting the keys valur pair
print("Summary :- ", result['summary'])
print("Sentiment :- ", result['sentiment'])
print("Key Themes :- ", result['key_themes'])
