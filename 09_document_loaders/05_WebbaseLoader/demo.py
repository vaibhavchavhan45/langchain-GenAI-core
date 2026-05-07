from langchain_community.document_loaders import WebBaseLoader

# e.g. 1
url = "https://www.bbc.com/news/topics/ce1qrvleleqt"
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs)) # 1
print(docs) # [Document(metadata = {}, page_content = '')]
print(type(docs)) # list 


# e.g. 2 WebBaseLoader is capable of processing multiple urls

url_1 = "https://www.bbc.com/news/topics/ce1qrvleleqt"
url_2 = "https://www.bbc.com/news/articles/cd6xz12j6pzo"
url_3 = "https://www.bbc.com/news/articles/ce3wyplnev1o"

loader = WebBaseLoader(url_1, url_2, url_3)

docs = loader.load()

print(len(docs)) # 3


# Note :
# If webpage is JS heavy means everything works on clicks then use SeleniumURLLoader
# For static pages (e.g. news, article, papers etc.) use WebBaseLoader