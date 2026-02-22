import os
from typing import List

from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX environment variable not set")

_pc = Pinecone(api_key=PINECONE_API_KEY)

_index = _pc.Index(PINECONE_INDEX_NAME)

_embedding_model = "llama-text-embed-v2"


def get_embeddings():
    return PineconeEmbeddings(model=_embedding_model)


def get_index():
    return _index


def add_documents(documents: List[dict], namespace: str = ""):
    embeddings = get_embeddings()
    texts = [doc["text"] for doc in documents]
    text_embeddings = embeddings.embed_documents(texts)

    vectors = []
    for i, doc in enumerate(documents):
        metadata = doc.get("metadata", {})
        metadata["text"] = doc.get("text", "")
        vectors.append(
            {
                "id": doc["id"],
                "values": text_embeddings[i],
                "metadata": metadata,
            }
        )

    _index.upsert(vectors=vectors, namespace=namespace)


def delete_documents(ids: List[str], namespace: str = ""):
    _index.delete(ids=ids, namespace=namespace)


def similarity_search(query_text: str, top_k: int = 4, namespace: str = "") -> List[dict]:
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(index=_index, embedding=embeddings)

    docs = vector_store.similarity_search(query_text, k=top_k, namespace=namespace)

    return [
        {
            "id": doc.metadata.get("id", ""),
            "text": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def get_index_info():
    index_desc = _pc.describe_index(PINECONE_INDEX_NAME)
    return {
        "index_name": PINECONE_INDEX_NAME,
        "dimension": index_desc.dimension,
        "metric": index_desc.metric,
        "embedding_model": _embedding_model,
        "endpoint": f"https://{index_desc.host}",
    }


def ensure_index():
    existing_indexes = _pc.list_indexes()
    if PINECONE_INDEX_NAME not in [idx["name"] for idx in existing_indexes]:
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
