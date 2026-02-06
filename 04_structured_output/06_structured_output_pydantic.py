from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv("GROQ_API_KEY"),
    openai_api_base = "https://api.groq.com/openai/v1"
)

class review(BaseModel):
    key_points: list[str] = Field(description = "Write down all the key points provided in review by the user")
    summary: str = Field(description = "Write down the in brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description = "Write down the sentiment based on the review as < positive | negative >")
    pros: Optional[list[str]] = Field(description = "Fetch all the positive points from review")
    cons: Optional[list[str]] = Field(description = "Fetch all the negative points from review")
    name: Optional[str] = Field(description = "Mention the name of the reviewer")

structured_model = model.with_structured_output(review)

result = structured_model.invoke(
   """I recently flew Emirates Airlines from Dubai to New York on their A380, and the experience was genuinely exceptional. I booked their Business Class, and from the moment I stepped into the airport, everything felt premium and well-organized.
The check-in process was smooth—dedicated counters with minimal wait time, and the staff were incredibly polite and efficient. The Emirates lounge at Dubai International was fantastic: spacious seating, a wide variety of food (both international and Middle Eastern cuisine), quiet zones, and even shower facilities to freshen up before the flight.
Onboard, the Business Class cabin was spacious with lie-flat seats that actually converted into comfortable beds—I managed to get 6 hours of solid sleep on a 14-hour flight, which is rare for me. The in-flight entertainment system had over 5,000 channels, including the latest movies, TV shows, and music. The touchscreen was responsive, and noise-canceling headphones were provided.
The food service was impressive—multiple courses with gourmet options, and the crew came around frequently to ensure everything was perfect. The Arabic coffee and dates were a nice cultural touch. The cabin crew were attentive without being intrusive, always greeting passengers with a smile and responding quickly to requests.
However, there were a few downsides. The Wi-Fi, while available, was painfully slow and expensive ($20 for 150MB is ridiculous in 2024). Also, despite booking months in advance, I couldn't select my preferred seat online—I had to call customer service, which took 40 minutes on hold.
The amenity kit was nice but felt a bit generic compared to what other premium airlines offer. And while the flight landed on time, baggage claim took nearly an hour, which was frustrating after such a long journey.

Pros:
Exceptional cabin crew service and hospitality
Comfortable lie-flat seats with excellent bedding
Outstanding in-flight entertainment and dining
Luxurious airport lounge with great amenities

Cons:
Overpriced and sluggish Wi-Fi connectivity
Limited online seat selection options
Long baggage claim wait times
Amenity kit could be more premium

Reviewed by John Doe
"""
)

print(result)
print("Key Points :- " , result.key_points)
print("Summary :- ", result.summary)
print("Sentiment :- ", result.sentiment)