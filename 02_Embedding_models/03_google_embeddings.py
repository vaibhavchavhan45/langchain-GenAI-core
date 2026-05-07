from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Google Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    # model="models/text-embedding-004",
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Test
text = "Hello, this is a test"
result = embeddings.embed_query(text)

print(f"Text: {text}")
print(f"Embedding dimension: {len(result)}")
print(f"First 10 values: {result}")