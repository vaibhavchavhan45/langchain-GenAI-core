from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = 'social_network_ads.csv')

docs = loader.lazy_load()

for item in docs:
    print(item.page_content)
