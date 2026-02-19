from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# load the docs
loader = TextLoader("docs.txt")
docs = loader.load()

# split the docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
chunks = text_splitter.split_documents(docs)

# create embeddings
embeddings = OpenAIEmbeddings()

# stores in db
vector_store = FAISS.from_documents(chunks, embeddings)

# retrieval
retriever = vector_store.as_retriever()

# llm
llm = OpenAI(
    model_name = 'gpt-3.5-turbo',
    temperature = 0.7
)

# create retrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "What are the key takeways from this document?"

result = qa_chain.invoke({'query' : query})
# OR
# result = qa_chain.run(query)  # .run() deprecated but works

print("Answer ", result)
