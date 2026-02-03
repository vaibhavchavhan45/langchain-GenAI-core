from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are an helpful business customer support AI'),
    MessagesPlaceholder(variable_name = 'h'),
    ('user', '{query}')
])

chat_history = []

with open('history.txt') as rl:
    chat_history.extend(rl.readlines())

prompt = chat_template.invoke({
    'h' : chat_history,
    'query' : 'Where is my refund'
})

print(prompt)
