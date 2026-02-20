# Problem statement
# Generate a tweet and linkeden post from the single prompt (parallel execution)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

templateX = PromptTemplate(
    template = 'Write a tweet for the following topic {topic}',
    input_variables = ['topic']
)

templateLinkeden = PromptTemplate(
    template = 'Write a linkeden post for the following topic {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet' : RunnableSequence(templateX, model, parser),
    'linkeden' : RunnableSequence(templateLinkeden, model, parser)
})

result = chain.invoke({'topic' : "Block-chain"})

print(result)
print("==============================")
print("Tweet : ", result['tweet'])
print("===============================")
print("linkeden Post : ", result['linkeden'])

# feat: add parallel LLM flow to create tweet and LinkedIn post from same topic
