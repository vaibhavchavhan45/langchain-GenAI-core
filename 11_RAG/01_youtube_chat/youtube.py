from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableSequence, 
    RunnableParallel, 
    RunnableLambda, 
    RunnablePassthrough
)
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Video loading
video_id = "SwQhKFMxmDY"
try: 
    ytt_transcript = YouTubeTranscriptApi()
    all_transcript =  ytt_transcript.fetch(video_id, languages = ["en"])

    transcript = " ".join(item.text for item in all_transcript)

except TranscriptsDisabled:
    print("No caption for this video")

# splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = text_splitter.create_documents([transcript])

# embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model = "models/text-embedding-004"
)

# store those embeddings
vector_store = FAISS.from_documents(
    chunks,
    embedding
)

# retrival of text
retriever = vector_store.as_retriever(
    search_type = "similarity", 
    search_kwargs = {"k" : 5}
)

# prompt
template = PromptTemplate(
    template = """
You are an assistant that answers questions strictly using the provided transcript context.

Rules:
- Use ONLY the information from the context.
- Do NOT add external knowledge.
- If the answer is not present in the context, respond with: "I don't know OR Question out of Scope"

Context:
{context}

Question:
{query}
""",
input_variables = ['context', 'query']
)

# function to combine retriever o/p documents
def format_text_context(input_text):
    context_text = "\n\n".join(item.page_content for item in input_text)
    return context_text

# forming a parallel chain
parallel_chain = RunnableParallel({
    "query" : RunnablePassthrough(),
    "context" : RunnableSequence(retriever, RunnableLambda(format_text_context))
})

# llm
llm = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

# output parser
parser = StrOutputParser()

# prompt--> model --> parsing output
chain = template | llm | parser

# combining parallel and simple chain
final_chain = RunnableSequence(parallel_chain, chain)

# Getting the output from llm
result = final_chain.invoke("Generate the summary of this video")

print(result)