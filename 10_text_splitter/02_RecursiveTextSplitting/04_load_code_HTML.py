from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("source.html")
code = loader.load()

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.HTML,
    chunk_size = 200,
    chunk_overlap = 0
)

chunk = splitter.split_documents(code)

print(chunk[0])