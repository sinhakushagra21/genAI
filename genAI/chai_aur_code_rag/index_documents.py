from langchain_community.document_loaders import WebBaseLoader
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

from create_vector_embeddings import createEmbeddings

load_dotenv()

WEBSITE_NAME = "chai_aur_code"

def fetchUrls(website_name):
    try:
        with open("genAI/chai_aur_code_rag/web_urls.json","r") as f:
            data = json.load(f)

        # list of urls
        if website_name in data:
            urls = data[website_name]
            return urls
        else:
            return KeyError("invalid website given:", website_name)
    except Exception as e:
        return Exception("error in fetching urls.", e)

def splitDocuments(documents):
    """split the documents into chunks"""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )

        docs = splitter.split_documents(documents=documents)

        return docs
    except Exception as e:
        return Exception("error in splitting documents", e)



def storeSplittedDocs(docs, embeddings, url, collection_name):
    """ store the docs into vector db"""
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url = url,
        collection_name = collection_name
    )

    return vector_store


def indexDocuments():
    # fetch the urls
    urls = fetchUrls(WEBSITE_NAME)

    # pass it to webParser(webBaseLoader)
    try:
        loader = WebBaseLoader(urls)
        documents = loader.load()
    except Exception as e:
        return Exception("error in WebBase Loader", e)
    
    # split the urls into chunks
    splitted_docs = splitDocuments(documents=documents)


    # # create vector embeddings
    embeddings = createEmbeddings()

    # using [embedding] model store the splitted_docs into vector db
    vector_store = storeSplittedDocs(
        splitted_docs,
        embeddings,
        "http://localhost:6333",
        "rag_system"
    )

    print("indexing of documents done.....")



if __name__ == "__main__":
    indexDocuments()
    
    









