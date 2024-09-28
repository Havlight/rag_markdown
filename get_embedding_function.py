from langchain_community.embeddings import JinaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    # embeddings = JinaEmbeddings(
    #     model_name="jina-embeddings-v2-base-zh",
    # )
    embeddings = OllamaEmbeddings(model="bge-m3")

    return embeddings
