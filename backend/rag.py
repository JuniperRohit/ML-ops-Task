import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass


@dataclass
class Document:
    page_content: str
    metadata: dict

USE_LOCAL = os.environ.get("ROHIT_USE_LOCAL_VECTORSTORE", "1") == "1"

if not USE_LOCAL:
    import pinecone
    from langchain.vectorstores import Pinecone


DATA_FOLDER = Path(__file__).resolve().parent.parent / "knowledge"
INDEX_NAME = os.environ.get("ROHIT_PINECONE_INDEX", "mlops-knowledge")


class LocalVectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.documents: List[Document] = []

    def embed(self, text: str):
        return self.model.encode(text, convert_to_numpy=True)

    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)

    def similarity_search(self, query: str, top_k: int = 4):
        query_emb = self.embed(query)
        scores = []
        for doc in self.documents:
            doc_emb = self.embed(doc.page_content)
            sim = float((query_emb @ doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-10))
            scores.append((sim, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]


class PineconeVectorStore:
    def __init__(self, api_key: str, environment: str, index_name: str = INDEX_NAME):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=self.model.encode("test").shape[0])
        self.index = pinecone.Index(index_name)

    def add_documents(self, docs: List[Document]):
        vectors = []
        for i, doc in enumerate(docs):
            vectors.append((str(i), self.model.encode(doc.page_content), doc.metadata))
        self.index.upsert(vectors=vectors)

    def similarity_search(self, query: str, top_k: int = 4):
        query_embedding = self.model.encode(query)
        res = self.index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
        return [Document(page_content=item["metadata"].get("source", ""), metadata=item["metadata"]) for item in res["matches"]]


def get_vector_store():
    if USE_LOCAL:
        if not hasattr(get_vector_store, '_instance'):
            get_vector_store._instance = LocalVectorStore()
        return get_vector_store._instance

    api_key = os.environ.get("ROHIT_PINECONE_API_KEY")
    env = os.environ.get("ROHIT_PINECONE_ENV")
    if not api_key or not env:
        raise ValueError("Pinecone credentials not configured")
    return PineconeVectorStore(api_key=api_key, environment=env)


def load_knowledge_from_folder(folder: Optional[Path] = None):
    folder = folder or DATA_FOLDER
    store = get_vector_store()
    docs: List[Document] = []

    for path in folder.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))

    for path in folder.rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))

    store.add_documents(docs)
    return len(docs)


def retrieve_context(question: str, top_k: int = 4):
    store = get_vector_store()
    results = store.similarity_search(question, top_k=top_k)
    return "\n\n".join([f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in results])
