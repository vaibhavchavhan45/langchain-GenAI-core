from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional

load_dotenv()

# ALl models of hugging face doesn't supports the with_structured_output (Recommendation: Use OutputParser then)
llm = HuggingFaceEndpoint(
    repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation'       
)

model = ChatHuggingFace(
    llm = llm
)

class Review(BaseModel):
    key_points: list[str] = Field(description = "Mention all the key points presented by the user in review")
    summary: str = Field(description = "Write down the summary of the review in brief")
    sentiment: Literal["pos", "neg"] = Field(description = "Write the sentiment of the review as < positive | negative >")
    pros: Optional[list[str]] = Field(description = "Write all the positive points from the review")
    cons: Optional[list[str]] = Field(description = "Write all the negative points from the review")
    name: Optional[str] = Field(description = "Mention the name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """I recently switched my internet provider to Airtel Xstream Fiber's 200 Mbps plan after dealing with Jio's pathetic customer service for months. The installation was quick and efficient—the technician came within two days of booking, routed cables cleanly along the walls, and configured everything in under an hour. The connection quality has been excellent with stable speeds throughout the day, and I'm finally able to attend Zoom meetings without constantly freezing or dropping out. The bundled Airtel Xstream app with free content is a decent bonus, though I rarely use it since I already have my own subscriptions.

    However, the pricing feels slightly higher than competitors at ₹1,099 per month for similar speeds, and they charge an additional ₹1,500 security deposit upfront which feels unnecessary. The router they provided is basic and doesn't support WiFi 6, which is disappointing for a premium plan. On the positive side, their customer service has been far more responsive than Jio—I raised a complaint about slow evening speeds once, and they sent a technician the next day who fixed it immediately. Overall, Airtel delivers reliable connectivity without the headache of dealing with horrible support, which alone makes the switch worth it for me.
    Review by Alice Wellington
    """
)

print(result)