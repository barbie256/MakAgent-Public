"""Offline-only sentence-transformer embedding runtime."""

from functools import lru_cache
import logging
import os
from pathlib import Path
from threading import Lock

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BACKEND_ROOT / "models" / MODEL_NAME
REQUIRED_MODEL_FILES = ("config.json", "modules.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILES = ("tokenizer.json", "vocab.txt")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger("uvicorn.error")
_MODEL = None
_MODEL_PATH: Path | None = None
_MODEL_LOCK = Lock()


class EmbeddingModelUnavailable(RuntimeError):
    """Raised immediately when the local embedding model cannot be used."""


def resolve_embedding_model_path(configured_path: str | os.PathLike | None = None) -> Path:
    raw_path = configured_path or os.getenv("RAG_EMBEDDING_MODEL_PATH")
    path = Path(raw_path).expanduser() if raw_path else DEFAULT_MODEL_PATH
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    return path.resolve()


def validate_local_model_path(path: Path) -> None:
    if not path.is_dir():
        raise EmbeddingModelUnavailable("Embedding model is not provisioned locally.")
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).is_file()]
    if not any((path / name).is_file() for name in WEIGHT_FILES):
        missing.append("model weights")
    if not any((path / name).is_file() for name in TOKENIZER_FILES):
        missing.append("tokenizer files")
    if missing:
        raise EmbeddingModelUnavailable(
            "Embedding model is not provisioned locally. Missing: " + ", ".join(missing)
        )


class EmbeddingEngine:
    """Thread-safe, globally reusable, offline-only embedding engine."""

    def __init__(self, model_path: str | os.PathLike | None = None):
        self.model_path = resolve_embedding_model_path(model_path)
        self.dimension = EMBEDDING_DIMENSION

    def _get_model(self):
        global _MODEL, _MODEL_PATH
        if _MODEL is None or _MODEL_PATH != self.model_path:
            with _MODEL_LOCK:
                if _MODEL is None or _MODEL_PATH != self.model_path:
                    validate_local_model_path(self.model_path)
                    try:
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer(os.fspath(self.model_path), local_files_only=True)
                        dimension_getter = getattr(model, "get_embedding_dimension", None) or getattr(
                            model, "get_sentence_embedding_dimension"
                        )
                        model_dimension = dimension_getter()
                        if model_dimension != self.dimension:
                            raise EmbeddingModelUnavailable(
                                f"Expected {self.dimension} embedding dimensions, received {model_dimension}"
                            )
                    except EmbeddingModelUnavailable:
                        raise
                    except Exception as exc:
                        raise EmbeddingModelUnavailable(
                            "Embedding model is not provisioned locally."
                        ) from exc
                    _MODEL = model
                    _MODEL_PATH = self.model_path
                    logger.info("RAG embedding model ready path=%s dimension=%s", self.model_path, self.dimension)
        return _MODEL

    def readiness_check(self) -> dict:
        self._get_model()
        return {"ready": True, "path": os.fspath(self.model_path), "dimension": self.dimension}

    def embed_documents(self, texts: list[str]):
        import numpy as np
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self._get_model().encode(
            texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str):
        return self.embed_documents([query])[0]


@lru_cache(maxsize=1)
def get_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()
