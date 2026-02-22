# Problem statement
# Dynamically select the appropriate LLM for a task based on its complexity to reduce cost without sacrificing quality.
# OR Reduce LLM cost by routing requests to different models based on task complexity.

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableLambda, 
    RunnableBranch, 
    RunnableSequence
)
import os

load_dotenv()

# Heavy LLM
heavy_model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1',
    temperature = 0.7
)

# Light LLM
light_model = ChatOpenAI(
    model = 'openai/gpt-oss-20b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1',
    temperature = 0.7
)

# python fucn to check complexity score
def check_complexity_score (text: str) -> int:
    score = 0
    input_text_to_lowercase = text.lower()

    # text based scoring
    if len(text) > 500 :
        score += 4
    elif len(text) > 300 :
        score += 3
    elif len(text) > 100 :
        score += 1
    
    # reasoning keyword based scoring
    reasoning_keywords = [
        "analyze", "design", "architecture", "optimize",
        "compare", "tradeoff", "scalability", "system"
    ]
    if any (k in input_text_to_lowercase for k in reasoning_keywords):
        score += 4
    
    # output depth indicator
    output_depth_indicator = [
        "step by step", "detailed", "explain", "breakdown", "report"
    ]
    if any(k in input_text_to_lowercase for k in output_depth_indicator):
        score += 3

    # simple task
    simple_task = [
        "summarize", "rewrite", "grammar", "fix", "one-liner", "short", "brief"
    ]
    if any(k in input_text_to_lowercase for k in simple_task):
        score -= 2

    return score

# convert that complexity score check to runnable
complexity_runnable = RunnableLambda(
    lambda text: {
        'text' : text,
        'score' : check_complexity_score(text)
    }
    
)

# score checking runnable
def heavy_task(data: dict) -> bool:
    return data["score"] >= 5

heavy_task_runnable = RunnableLambda(heavy_task)

# Based on the task assign LLM
branch_chain = RunnableBranch(
    (
        heavy_task_runnable,
        RunnableLambda(lambda x: x["text"]) | heavy_model
    ),
    RunnableLambda(lambda x: x['text']) | light_model
)

# first check complexity score then send the prompt to LLM according to score
chain = RunnableSequence(complexity_runnable, branch_chain)

result = chain.invoke(
    "Design a scalable backend architecture for a real-time chat application"
)

print(result)