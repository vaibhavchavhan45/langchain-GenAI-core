from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("deep-learning-docs.pdf")

docs = loader.load()

# print(docs)
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)


# Lazy Load
# document = loader.lazy_load()

# for page in document:
#     print(page.page_content)

# Note : 
# load --> Loads all the documents in memoryview
# lazy_load --> Loads one document at a single time then remove it loads next and so on.....