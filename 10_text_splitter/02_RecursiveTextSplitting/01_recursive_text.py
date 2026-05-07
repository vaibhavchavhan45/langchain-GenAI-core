from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
    Morning sunlight spills across quiet streets, warming the walls of old buildings and waking the city slowly. People step outside with cups of tea, conversations starting softly as the day begins to move. The air feels calm, carrying the promise of routine and small moments of comfort.

    As evening approaches, the same streets grow louder and more alive with footsteps and voices. Shops glow under artificial lights, and tired faces relax after a long day. The city settles into a different rhythm, preparing itself for another night of rest and reflection.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

result = splitter.split_text(text)

print(result)