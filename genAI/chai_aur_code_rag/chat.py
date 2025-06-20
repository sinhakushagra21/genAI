
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

from create_vector_embeddings import createEmbeddings

load_dotenv()
client = OpenAI()

def chatBot():
    # create vector embedding of the query
    embeddings = createEmbeddings()

    vector_db = QdrantVectorStore.from_existing_collection(
        collection_name="rag_system",
        embedding=embeddings,
        url="http://localhost:6333"
    )
    print("ChatBot is ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        # take user query
        query = input("> ").strip()

        if query.lower() in ["exit", "quit"]:
            print("[BOT] : Goodbye!")
            break

        # vector search
        search_result = vector_db.similarity_search(query=query)

        # prepare context for LLM
        context = "\n\n\n".join(
            [f"Page contents: {result.page_content}\n\nWeb page: {result.metadata['source']}" for result in search_result]
        )

        # Provide context(search result) + user query to LLM and using system prompt answer the query.
        SYSTEM_PROMPT = f"""
            You are a helpful AI assistant. You have to answer user query based on the availaible context retrieved
            from the web scrapping(provided) along with web content and web url.

            If asked anything else apart from the context, please kindly deny to answer.

            Example:
            User: How to ride bike?
            Assistant: Sorry!! I cant answer the question as it is not related.....

            You should only answer the user based on the following context and navigate the user to open the right
            web page to know more.

            Context : 
            {context}
            """
    
        chat_completions = client.chat.completions.create(
            model='gpt-4.1',
            messages=[
                {
                    "role": "system", "content": SYSTEM_PROMPT
                },
                {
                    "role": "user", "content": query
                }
            ]
        )

        print(f"[BOT] : {chat_completions.choices[0].message.content.strip()}")

if __name__ == "__main__":
    chatBot()