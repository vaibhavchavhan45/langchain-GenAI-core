from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = 'Books',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

# lazy load --> return generator
docs = loader.lazy_load()

for item in docs:
    print(item.metadata)


# load --> returns list
docs = loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)


# Note :
# 1. single PDF → PyPDFLoader
# 2. Multiple PDFs (folder) → DirectoryLoader + PyPDFLoader