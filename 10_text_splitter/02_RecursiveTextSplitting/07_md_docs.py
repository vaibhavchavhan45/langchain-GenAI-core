from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("markdown.md")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,
    chunk_size = 200,
    chunk_overlap = 0
)

chunk = splitter.split_documents(docs)

print(chunk[0].page_content)
print(chunk[0].metadata)