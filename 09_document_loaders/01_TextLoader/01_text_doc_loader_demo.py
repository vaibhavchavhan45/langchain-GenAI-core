# TextLoader is deprecated in newer version of langchain
# So, instead use python's syntax to load the document

from langchain_community.document_loaders import TextLoader

loader = TextLoader("01_cric.txt", encoding='utf-8')

docs = loader.load()

print(docs) # [Document(metadata: {01_cric.txt}, page_content='WHOLE CONTENT')]
print(type(docs)) # list/array (for consistent api)
print(len(docs))  # 1
print(type(docs[0])) # langchain_core.documents.base.Document
print(docs[0].page_content) # whole text output
print(type(docs[0].page_content)) # str
print(docs[0].metadata)  # consist of following things not exactly but roughly
# {
#   "source": "a.txt",
#   "file_path": "/full/path/a.txt",
#   "page": 1,
#   "chunk": 0
# }
print(type(docs[0].metadata))  # dict