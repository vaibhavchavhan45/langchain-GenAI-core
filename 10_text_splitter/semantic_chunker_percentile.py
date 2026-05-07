from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="embed-english-light-v3.0"
)

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
)

docs = splitter.create_documents([sample])
print(docs)
print(len(docs))