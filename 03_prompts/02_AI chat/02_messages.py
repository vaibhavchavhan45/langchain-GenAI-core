from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = "https://api.groq.com/openai/v1"
)

messages = [
    SystemMessage(content = "You are helpful and informative AI assistant, has immense knowledge about literature and related authors"),
    HumanMessage(content = "In brief information about nobel price awardee in literature and reference of that literature for that he/she won the nobel price")
]

result = model.invoke(messages)
messages.append(AIMessage(content = result.content))

print(messages)