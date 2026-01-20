from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='Gemini 2.5 Flash'
)

result = model.invoke("Where is India located?")

print(result.content)