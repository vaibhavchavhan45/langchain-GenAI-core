from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

large_language_model = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = 'text-generation'
)

model = ChatHuggingFace(llm=large_language_model)

result = model.invoke("What is the capital of India.")

print(result.content)