from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

#schema
json_schema = {
    "title" : "Review",
    "description" : "The review for user prompt",
    "type": "object",
    "properties" : {
        "key_points" : {
            "title" : "key_points",
            "description" : "Mentioned key point",
            "type" : "array",
            "items" : {
                "type" : "string"
            }
        },
        "summary" : {
            "title" : "summary",
            "description" : "brief summary for review",
            "type" : "string"
        },
        "sentiment" : {
            "title" : "sentiment",
            "description" : "sentiment of review as positive or negative",
            "enum" : ["pos", "neg"]
        },
        "pros" : {
            "title" : "pros",
            "description" : "positive point of review",
            "type" : ["array", "null"],
            "items" : {
                "type" : "string" 
            }
        },
        "cons" : {
            "title" : "cons",
            "description" : "negative point of review",
            "type" : ["array", "null"],
            "items" : {
                "type" : "string"
            }
        },
        "name" : {
            "title" : "name",
            "description" : "name of the reviewer",
            "type" : ["string", "null"]
        }
    },
    "required" : ["key_points", "summary", "sentiment"]
}

structured_output = model.with_structured_output(json_schema)

result = structured_output.invoke("""
    I recently started using Swiggy Instamart for my weekly grocery shopping after getting tired of crowded supermarkets. The app interface is clean and easy to navigate—I can find everything from vegetables to snacks within minutes, and the search function actually works well. Deliveries have been impressively quick, usually arriving within 15-20 minutes of placing the order, which is perfect when I realize I'm out of milk at 10 PM. The packaging is decent with separate bags for different items, and I appreciate that they don't mix cleaning products with food items like some other services do.

However, the prices are noticeably higher than local stores—I'm paying almost 20-30% more for the same brands, which adds up quickly over a month. The product availability is inconsistent; items showing "in stock" often become unavailable right when I'm checking out, forcing me to reorder or find alternatives. The delivery executives sometimes leave orders at the building gate instead of bringing them to my door, despite clear instructions in the app. On the positive side, their customer support is responsive—when I received spoiled tomatoes once, they refunded the amount within an hour without any hassle. The subscription model
""")

print(result)
print("Key points :- ", result['key_points'])
print("Summary :- ", result['summary'])
print("Sentiment :- ", result['sentiment'])