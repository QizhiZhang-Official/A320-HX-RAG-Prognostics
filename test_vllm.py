from langchain_openai import ChatOpenAI


model = ChatOpenAI(
    model='/model',
    base_url='http://localhost:8000/v1',
    api_key='EMPTY',
    temperature=0.1
)

messages = [
    (
        "system",
        "You are a helpful translator. Translate the user sentence to Japanese.",
    ),
    ("human", "I love programming."),
]
print(model.invoke(messages).content)