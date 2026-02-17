from langchain.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

llm = OpenAI(
    model_name = "gpt-3.5-turbo",
    temperature = 0.7
)

# Load the document
loader = TextLoader("docs.txt") # tracks the file
documents = loader.load() # load the actual content

# splits the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50) 
    # Each chunk is of 500 characters and 
    # chunk overlap means Last 50 chars of previous chunk repeat in next chunk (maintains context for LLM)
    # e.g.
    # docs : ABCDEFGHIJKL..... upto 1000
    # chunk 1 : A to 500th character
    # chunk 2 : 451 to 950th character (last 50 of Chunk 1 + next 450 new)
    # chunk 3 : 901 to 1000th character (last 50 of chunk 2 + next remaining all)
docs = text_splitter.split_documents(documents) # docs containing all chunks in array (array name -> docs)

# converts text into embeddings and stores in db
vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    # For setup ==> OpenAIEmbeddings generates embeddings and FAISS stores that embeddings in vector db
    # For user query ==> OpenAIEmbeddings generates embeddings and FAISS searches the 5 relative chunk in vector db when retriver tells FAISS to do the search and returns (vectors + text) to retriver

# retriver
retriever = vector_store.as_retriever() # Creates retriever object (setup only, no search yet)

query = "What are the key takeaways from this documents?"

retrieved_docs = retriever.get_relevant_documents(query) # Uses FAISS to search, gets top 5 matching chunks (text only) as Document objects in a list

# combine retrived text into single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs]) # combine all 5 responses from the retrived_docs as a single string and separates each single response out of those 5 as a new line

# manually pass retrived text to LLM
prompt = f"Based on the following text answer the questions : {query} \n\n {retrieved_text}"

result = llm.invoke(prompt)

print("Answer >> ", result)

