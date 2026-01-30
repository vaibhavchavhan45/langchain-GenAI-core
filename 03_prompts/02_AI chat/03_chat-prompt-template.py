from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate(
    [
        ('system', 'You are an smart AI assistant, expert in {domain}'),
        ('user', 'Explain about the {topic} in the {domain}' )
    ]
)

prompt = chat_template.invoke({
    'domain' : 'cricket',
    'topic' : 'doosra'
})

print(prompt)

