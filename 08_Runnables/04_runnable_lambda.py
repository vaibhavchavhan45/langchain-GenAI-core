                    # Problem statement
# Built a joke generator also validate the joke should be less than 20 words

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

template = PromptTemplate(
    template = 'Create a joke on the given provided topic {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

joke_generation_chain = RunnableSequence(template, model, parser)

# traditional way (preferable)
def word_counter(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_counter)
})

# # modern way
# parallel_chain = RunnableParallel({
#     'joke' : RunnablePassthrough(),
#     'word_count' : RunnableLambda(lambda x : len(x.split()))
# })

chain = RunnableSequence(joke_generation_chain, parallel_chain)

result = chain.invoke({'topic' : 'AI'})

# final_result = """ {} \n word count : """.format(result['joke'], result['word_count'])
# OR
final_result = f"{result['joke']} \n Word Count : {result['word_count']}"

print(final_result)
####### feat: implement joke generation pipeline with non-enforced length validation

# validation means : Only we defined prechecks for the llm response if the response is less than 20 words means it will return true and if it's not then it will return false
# Error handeling according to response as true or false on frontend we do other checks like should this response to be display or not on UI