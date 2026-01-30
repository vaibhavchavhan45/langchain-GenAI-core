from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv("GROQ_API_KEY"),
    openai_api_base = "https://api.groq.com/openai/v1"
)

messages = [
    SystemMessage(content='You are an smart AI assistant')
]

while True:
    user_input = input("User >> ")
    messages.append(HumanMessage(content = user_input))
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    messages.append(AIMessage(content = result.content))
    print("AI >> ", result.content)

print(messages)