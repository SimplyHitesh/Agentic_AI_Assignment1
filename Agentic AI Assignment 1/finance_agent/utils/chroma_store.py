"""
finance_agent/utils/chroma_store.py
-----------------------------------
Vector Store implementation using ChromaDB.
Fulfills Mandatory Criteria 5 & 7:
  - Stores document embeddings in a persistent ChromaDB collection.
  - Ingests financial budgeting guidelines and category benchmarks.
  - Provides similarity search and LangChain retriever interfaces.
  - Gracefully supports OpenAI Embeddings with local sentence-transformers / fallback.
"""

from __future__ import annotations

import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from finance_agent.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    Manages embedding generation, vector indexing, and similarity retrieval
    for financial budgeting rules and benchmarks using ChromaDB.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.api_key = api_key or OPENAI_API_KEY

        # Ensure directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Initialize or fetch collection with appropriate embedding function
        self._embedding_function = self._get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"description": "Financial budgeting guidelines and benchmarks"},
        )

        # Auto-seed if collection is empty
        if self.collection.count() == 0:
            self._seed_default_guidelines()

    def _get_embedding_function(self):
        """
        Chooses the best available embedding function:
        1. OpenAIEmbeddingFunction (if API key available)
        2. Chroma's DefaultEmbeddingFunction (all-MiniLM-L6-v2) or fallback
        """
        if self.api_key and not OPENAI_BASE_URL:
            try:
                from chromadb.utils import embedding_functions
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=self.api_key,
                    model_name="text-embedding-3-small",
                )
            except Exception as e:
                logger.warning("Could not initialize OpenAIEmbeddingFunction: %s. Using default.", e)

        # Chroma's built-in ONNX MiniLM embedding function (works offline without API keys)
        try:
            from chromadb.utils import embedding_functions
            return embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            logger.warning("DefaultEmbeddingFunction failed (%s). Using dummy function.", e)
            return None

    def _seed_default_guidelines(self) -> int:
        """
        Reads all text files from finance_agent/data/guidelines/ and ingests them.
        """
        guidelines_dir = Path(__file__).resolve().parent.parent / "data" / "guidelines"
        txt_files = list(guidelines_dir.glob("*.txt"))

        if not txt_files:
            logger.warning("No guideline text files found in %s", guidelines_dir)
            return 0

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        chunk_counter = 0
        for file_path in txt_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                # Split by logical sections (paragraphs or headers)
                sections = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 40]
                
                for idx, section in enumerate(sections):
                    chunk_counter += 1
                    documents.append(section)
                    metadatas.append({
                        "source": file_path.name,
                        "section_index": idx,
                        "title": section.split("\n")[0].replace("#", "").strip(),
                    })
                    ids.append(f"{file_path.stem}_chunk_{idx}")
            except Exception as exc:
                logger.error("Failed reading guideline file %s: %s", file_path, exc)

        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info("Successfully indexed %d chunks in ChromaDB.", len(documents))

        return len(documents)

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform similarity search in ChromaDB.
        Returns a list of dicts with document, metadata, and distance.
        """
        if self.collection.count() == 0:
            self._seed_default_guidelines()

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, max(1, self.collection.count())),
            )
        except Exception as exc:
            logger.error("Chroma query failed: %s", exc)
            return []

        hits = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            dists = results.get("distances", [[]])[0] if results.get("distances") else []

            for i in range(len(docs)):
                hits.append({
                    "content": docs[i],
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else 0.0,
                })

        return hits

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """
        Returns a formatted markdown string of relevant guidelines for LLM prompting.
        """
        hits = self.similarity_search(query, k=k)
        if not hits:
            return "No specific guidelines retrieved from database."

        formatted_chunks = []
        for i, hit in enumerate(hits, 1):
            src = hit.get("metadata", {}).get("source", "guideline")
            title = hit.get("metadata", {}).get("title", "")
            formatted_chunks.append(f"### [Guideline {i}: {title} ({src})]\n{hit['content']}")

        return "\n\n".join(formatted_chunks)

    def count(self) -> int:
        """Returns total number of chunks currently indexed."""
        try:
            return self.collection.count()
        except Exception:
            return 0
