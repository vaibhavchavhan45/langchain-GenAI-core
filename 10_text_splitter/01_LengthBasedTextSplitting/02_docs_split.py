from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("deep-learning-docs.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0,
    separator = "" 
)

result = splitter.split_documents(docs)

print(result[0])

# Structure
# [
#     Document(
#         page_content="chunk 1 text ...",
#         metadata={...}
#     ),
#     Document(
#         page_content="chunk 2 text ...",
#         metadata={...}
#     )  
# ]