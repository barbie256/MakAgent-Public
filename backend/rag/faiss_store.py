from dataclasses import dataclass, field
import re
from threading import RLock
from typing import Any

from rag.embeddings import EMBEDDING_DIMENSION, EmbeddingEngine, get_embedding_engine
from rag.faiss_persistence import load_index, save_index


FAQS_INDEX = "faiss_faqs_index"
POLICIES_INDEX = "faiss_policies_index"
SUPPORT_INDEX = "faiss_support_index"
FUTURE_INDEX = "faiss_future_index"

_CANONICAL_ALIASES = {
    "faiss_core_index": FAQS_INDEX,
    "faiss_student_index": POLICIES_INDEX,
    "faiss_support_index": SUPPORT_INDEX,
    "faiss_procedures_index": FUTURE_INDEX,
}


@dataclass(frozen=True)
class StoredDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FaissDocumentStore:
    """Thread-safe in-process FAISS index with separate document metadata."""

    def __init__(
        self,
        name: str,
        embedding_engine: EmbeddingEngine | None = None,
    ):
        self.name = name
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.dimension = EMBEDDING_DIMENSION
        self._index = None
        self._documents: list[StoredDocument] = []
        self._keyword_postings: dict[str, set[int]] = {}
        self._lock = RLock()
        self._version = 0
        self._restore_from_disk()

    def _restore_from_disk(self) -> None:
        """Hydrate persisted vectors and documents without changing public APIs."""
        try:
            index, payload = load_index(self.name)
        except ModuleNotFoundError:
            # Keep module imports usable in environments where optional FAISS
            # dependencies have not yet been installed.
            return
        if index is None:
            try:
                self._index = self._new_index()
            except ModuleNotFoundError:
                self._index = None
            return
        if getattr(index, "d", self.dimension) != self.dimension:
            raise ValueError(f"Persisted index dimension mismatch for {self.name}")

        documents = payload.get("documents", [])
        self._index = index
        self._documents = [
            StoredDocument(text=str(item["text"]), metadata=dict(item.get("metadata", {})))
            for item in documents
        ]
        self._version = int(payload.get("version", 0))
        self._rebuild_keyword_postings()

    def _rebuild_keyword_postings(self) -> None:
        self._keyword_postings = {}
        for position, document in enumerate(self._documents):
            for term in set(re.findall(r"[a-z0-9@.]+", document.text.lower())):
                self._keyword_postings.setdefault(term, set()).add(position)

    @staticmethod
    def _persistence_payload(
        documents: list[StoredDocument], version: int
    ) -> dict[str, Any]:
        return {
            "version": version,
            "documents": [
                {"text": document.text, "metadata": document.metadata}
                for document in documents
            ],
        }

    @property
    def version(self) -> int:
        return self._version

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._documents)

    def _new_index(self):
        import faiss

        return faiss.IndexFlatIP(self.dimension)

    def add_documents(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        import numpy as np

        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("Metadata count must match text count")

        input_metadata = metadata or [{} for _ in texts]
        candidates = [
            (text.strip(), dict(item))
            for text, item in zip(texts, input_metadata)
            if text and text.strip()
        ]
        with self._lock:
            known_ids = {
                document.metadata.get("document_id")
                for document in self._documents
                if document.metadata.get("document_id")
            }
            known_dedup_keys = {
                document.metadata.get("dedup_key")
                for document in self._documents
                if document.metadata.get("dedup_key")
            }

        unique_candidates = []
        for text, item in candidates:
            document_id = item.get("document_id")
            dedup_key = item.get("dedup_key")
            if (document_id and document_id in known_ids) or (dedup_key and dedup_key in known_dedup_keys):
                continue
            unique_candidates.append((text, item))
            if document_id:
                known_ids.add(document_id)
            if dedup_key:
                known_dedup_keys.add(dedup_key)

        if not unique_candidates:
            return 0
        clean_texts = [text for text, _ in unique_candidates]
        clean_metadata = [item for _, item in unique_candidates]

        vectors = self.embedding_engine.embed_documents(clean_texts)
        if vectors.shape != (len(clean_texts), self.dimension):
            raise ValueError("Embedding engine returned an inconsistent shape")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        with self._lock:
            if self._index is None:
                self._index = self._new_index()
                rollback_index = None
            else:
                import faiss

                rollback_index = faiss.clone_index(self._index)
            self._index.add(vectors)
            first_position = len(self._documents)
            new_documents = [
                StoredDocument(text=text, metadata=dict(item))
                for text, item in zip(clean_texts, clean_metadata)
            ]
            proposed_documents = [*self._documents, *new_documents]
            proposed_version = self._version + 1
            try:
                save_index(
                    self.name,
                    self._index,
                    self._persistence_payload(proposed_documents, proposed_version),
                )
            except Exception:
                self._index = rollback_index
                raise
            self._documents.extend(new_documents)
            for offset, text in enumerate(clean_texts):
                position = first_position + offset
                for term in set(re.findall(r"[a-z0-9@.]+", text.lower())):
                    self._keyword_postings.setdefault(term, set()).add(position)
            self._version = proposed_version
        return len(clean_texts)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0 or not query.strip():
            return []
        with self._lock:
            if self._index is None or not self._documents:
                return []

        import numpy as np

        query_vector = np.ascontiguousarray(
            self.embedding_engine.embed_query(query).reshape(1, -1),
            dtype=np.float32,
        )
        with self._lock:
            # Retrieve a wider semantic candidate set, then apply a lightweight
            # lexical rerank. This retains FAISS speed while improving exact-term
            # matches for university names, acronyms, and service terminology.
            result_count = min(max(top_k * 4, top_k), len(self._documents))
            scores, positions = self._index.search(query_vector, result_count)
            results = []
            query_terms = set(query.lower().split())
            for score, position in zip(scores[0], positions[0]):
                if position < 0:
                    continue
                document = self._documents[int(position)]
                document_terms = set(document.text.lower().split())
                lexical_score = (
                    len(query_terms.intersection(document_terms)) / len(query_terms)
                    if query_terms
                    else 0.0
                )
                vector_score = float(score)
                hybrid_score = 0.8 * vector_score + 0.2 * lexical_score
                results.append(
                    {
                        "text": document.text,
                        "score": hybrid_score,
                        "vector_score": vector_score,
                        "lexical_score": lexical_score,
                        "metadata": dict(document.metadata),
                    }
                )
            return sorted(
                results,
                key=lambda item: item["score"],
                reverse=True,
            )[:top_k]

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve exact-term candidates through an in-memory inverted index."""
        if top_k <= 0:
            return []
        query_terms = set(re.findall(r"[a-z0-9@.]+", query.lower()))
        if not query_terms:
            return []

        with self._lock:
            candidate_hits: dict[int, int] = {}
            for term in query_terms:
                for position in self._keyword_postings.get(term, ()):
                    candidate_hits[position] = candidate_hits.get(position, 0) + 1
            ranked_positions = sorted(
                candidate_hits,
                key=lambda position: candidate_hits[position],
                reverse=True,
            )[:top_k]
            return [
                {
                    "text": self._documents[position].text,
                    "score": candidate_hits[position] / len(query_terms),
                    "keyword_score": candidate_hits[position] / len(query_terms),
                    "metadata": dict(self._documents[position].metadata),
                }
                for position in ranked_positions
            ]


_stores = {
    name: FaissDocumentStore(name)
    for name in (FAQS_INDEX, POLICIES_INDEX, SUPPORT_INDEX, FUTURE_INDEX)
}


def get_store(name: str) -> FaissDocumentStore:
    name = _CANONICAL_ALIASES.get(name, name)
    try:
        return _stores[name]
    except KeyError as exc:
        raise ValueError(f"Unknown FAISS index: {name}") from exc


def get_all_stores() -> dict[str, FaissDocumentStore]:
    return dict(_stores)
