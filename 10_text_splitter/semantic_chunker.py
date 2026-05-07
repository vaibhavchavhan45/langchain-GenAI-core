from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

docs = splitter.create_documents([sample])

print(docs)
print(type(docs))

# Notes :
# 1. Splits text based on meaning/context rather than character count
# 2. Generates embeddings for each sentence and compares adjacent sentences
# 3. After comparing if there is similar context of the sentences then they will be in same chunk
# 4. If context changes means sudden drop in similarity then that sentence will go in next chunk 

### 4 breakpoint threshold type
# percentile (default value = 95%)
# standard_deviation (default value = 3)
# interquartile (default value = 1.5)
# gradient (default value = 95th percentile of gradient)