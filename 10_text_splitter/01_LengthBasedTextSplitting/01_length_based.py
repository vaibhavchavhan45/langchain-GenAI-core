from langchain_text_splitters import CharacterTextSplitter

text = """
    On a quiet morning, the city feels softer and more human, as if it is briefly taking a deep breath. Street vendors arrange their carts while birds argue loudly over crumbs near the sidewalk. The smell of fresh tea drifts from small shops, mixing with the cool air. People move slower, unhurried by deadlines or traffic. Sunlight reflects off windows, creating patterns that disappear as quickly as they appear. In these moments, even familiar places feel new. The day hasn’t demanded anything yet, and that calm feels like a small gift.
"""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = "." # can be comma, "", etc.
)

result = splitter.split_text(text)

print(result)

# It is consider that the chunk_overlap can be roughly in between 10% to 20% of chunk_size

# Note :
# chunk_size = 4000 (characters)
# chunk_overlap = 200 (characters)