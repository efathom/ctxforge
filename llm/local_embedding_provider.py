"""
Local (sentence-transformers) embedding provider.

Runs HuggingFace embedding models in-process via ``sentence-transformers``.
Intended for self-hosted / offline deployments. ``sentence-transformers`` is an
optional dependency (``pip install 'ctxforge[huggingface]'``).
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from ctxforge.engine.registry import registry
from ctxforge.protocols.llm import EmbeddingResponse, IEmbeddingProvider

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class LocalEmbeddingConfig:
    """Configuration for the local sentence-transformers embedding provider."""

    model: str = DEFAULT_LOCAL_MODEL
    device: Optional[str] = None  # None => auto (cuda if available else cpu)
    normalize_embeddings: bool = True
    batch_size: int = 32
    max_length: Optional[int] = None


class LocalEmbeddingProvider(IEmbeddingProvider):
    """
    In-process HuggingFace embedding provider (sentence-transformers).

    Accepts either a :class:`LocalEmbeddingConfig`, the framework's
    ``EmbeddingConfig`` (from YAML), or a plain dict of options.

    Example:
        from ctxforge.llm.local_embedding_provider import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(LocalEmbeddingConfig(model="BAAI/bge-small-en-v1.5"))
        emb = await provider.embed_single("hello")
    """

    def __init__(self, config: Any = None):
        cfg = self._coerce_config(config)
        self._model_name = cfg.get("model") or DEFAULT_LOCAL_MODEL
        self._device = cfg.get("device")
        self._normalize = bool(cfg.get("normalize_embeddings", True))
        self._batch_size = int(cfg.get("batch_size") or 32)
        self._max_length = cfg.get("max_length")
        self._dimension = int(cfg.get("dimension") or 0)
        self._model = None

    @staticmethod
    def _coerce_config(config: Any) -> dict:
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        out: dict = {}
        for key in (
            "model",
            "device",
            "normalize_embeddings",
            "batch_size",
            "max_length",
            "dimension",
        ):
            value = getattr(config, key, None)
            if value is not None:
                out[key] = value
        return out

    @property
    def name(self) -> str:
        return "local"

    @property
    def default_model(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return self._dimension

    async def get_dimension(self) -> int:
        """Return the model's embedding dimension, loading it if necessary.

        Used by the factory to auto-derive the vector-store dimension for local
        models (whose dimension differs from the OpenAI default).
        """
        await self._get_model()
        return self.embedding_dimension

    async def _get_model(self):
        if self._model is None:
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(None, self._load_model)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install 'ctxforge[huggingface]'"
            ) from None
        return SentenceTransformer(self._model_name, device=self._device)

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                model=model or self.default_model,
                total_tokens=0,
                latency_ms=0.0,
            )

        start = time.time()
        encoder = await self._get_model()
        loop = asyncio.get_running_loop()

        encode_kwargs: dict = {
            "batch_size": self._batch_size,
            "normalize_embeddings": self._normalize,
            "show_progress_bar": False,
        }
        if self._max_length:
            encode_kwargs["max_length"] = self._max_length
        encode_kwargs.update(kwargs)

        vectors = await loop.run_in_executor(
            None, lambda: encoder.encode(list(texts), **encode_kwargs)
        )

        return EmbeddingResponse(
            embeddings=[list(v) for v in vectors],
            model=model or self.default_model,
            total_tokens=sum(len(t.split()) for t in texts),
            latency_ms=(time.time() - start) * 1000,
        )

    async def embed_single(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        resp = await self.embed([text], model=model, **kwargs)
        return resp.embeddings[0] if resp.embeddings else []


# Register for config-driven wiring (``storage.memory.vector.embedding.provider: local``).
registry.register_embedding("local")(LocalEmbeddingProvider)
registry.register_embedding("sentence_transformers")(LocalEmbeddingProvider)
registry.register_embedding("huggingface")(LocalEmbeddingProvider)
