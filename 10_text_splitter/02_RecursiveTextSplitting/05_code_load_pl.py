from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

loader = GenericLoader.from_filesystem(
    ".", # "mnt/users/ASUS/Desktop..." (we can defined PATH in this format also)
    glob = "**/snippet.js",
    parser = LanguageParser(language = Language.JS)
)
docs = loader.load()

# splitter = RecursiveCharacterTextSplitter.from_language(
#     language = Language.JS, # if u use parser then this line is optional
#     chunk_size = 500,
#     chunk_overlap = 0
# )

# OR (if u use parser then either snippet works)
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0
)

chunk = splitter.split_documents(docs)

print(chunk[1].page_content)