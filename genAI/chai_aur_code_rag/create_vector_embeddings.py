from langchain_openai import OpenAIEmbeddings


def createEmbeddings():
    # create vector embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    return embeddings