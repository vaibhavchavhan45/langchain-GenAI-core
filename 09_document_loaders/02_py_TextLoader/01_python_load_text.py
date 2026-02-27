from langchain_core.documents import Document

with open("01_lang.txt", encoding='utf-8') as f:
    text = f.read()

docs = [
    Document(
    metadata = {"source": "01_lang.txt"}, 
    page_content = text
    )
]

print(docs)