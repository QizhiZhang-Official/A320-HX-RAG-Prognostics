from langchain_openai import ChatOpenAI


def build_chat_model(*, temprature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model="/model",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=temprature,
        streaming=True,
    )
