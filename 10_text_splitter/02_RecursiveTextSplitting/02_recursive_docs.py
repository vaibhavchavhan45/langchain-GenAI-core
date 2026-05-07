from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("deep-learning-docs.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

result = splitter.split_documents(docs)

print(result[0].page_content) # first chunk