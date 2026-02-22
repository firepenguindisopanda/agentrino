from typing import Optional, Any

import pinecone_service

DEFAULT_TOP_K = 4


class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def retrieve_context(query: str, top_k: int = DEFAULT_TOP_K) -> list[Document]:
    results = pinecone_service.similarity_search(query, top_k=top_k)
    return [Document(page_content=r["text"], metadata=r.get("metadata", {})) for r in results]


def format_context(documents: list[Document]) -> str:
    if not documents:
        return ""
    return "\n\n".join([doc.page_content for doc in documents])


def build_rag_prompt(
    query: str,
    system_prompt: Optional[str] = None,
    include_context: bool = True,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[str, list[Document]]:
    context_docs = []
    context_str = ""

    if include_context:
        context_docs = retrieve_context(query, top_k=top_k)
        context_str = format_context(context_docs)

    if context_str:
        full_prompt = f"""Context information:
{context_str}

Question: {query}"""
    else:
        full_prompt = query

    return full_prompt, context_docs
