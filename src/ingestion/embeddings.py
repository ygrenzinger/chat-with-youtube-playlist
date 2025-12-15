"""
Embedding strategies for text embedding generation.

Implements the Strategy pattern to allow easy switching between
different embedding providers (local models vs cloud APIs).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Base configuration for embedding strategies."""
    batch_size: int = 32
    max_length: int = 256


@dataclass
class LocalBGEM3Config(EmbeddingConfig):
    """Configuration for local BGE-M3 model."""
    use_fp16: bool = True
    device: str | None = None  # auto-detect if None


@dataclass
class DeepInfraConfig(EmbeddingConfig):
    """Configuration for DeepInfra API."""
    api_key: str = ""  # from env DEEPINFRA_API_KEY if empty
    model: str = "BAAI/bge-m3"
    base_url: str = "https://api.deepinfra.com/v1/openai"


class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            np.ndarray of shape (len(texts), dimension)
        """
        pass


class LocalBGEM3Embedding(EmbeddingStrategy):
    """
    Local BGE-M3 embedding using FlagEmbedding library.

    Runs the model locally on GPU/CPU. Model is lazily loaded
    on first use to avoid unnecessary memory allocation.
    """

    def __init__(self, config: LocalBGEM3Config | None = None):
        self.config = config or LocalBGEM3Config()
        self._model = None

    @property
    def model_name(self) -> str:
        return "BAAI/bge-m3"

    @property
    def dimension(self) -> int:
        return 1024

    def _get_model(self):
        """Lazy load the BGE-M3 model."""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"Loading local BGE-M3 model (fp16={self.config.use_fp16})...")
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.config.use_fp16,
            )
            logger.info("BGE-M3 model loaded")
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using local BGE-M3 model."""
        if not texts:
            return np.array([])

        model = self._get_model()
        logger.debug(f"Embedding {len(texts)} texts locally")

        result = model.encode(
            texts,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result["dense_vecs"]


class DeepInfraBGEM3Embedding(EmbeddingStrategy):
    """
    BGE-M3 embedding via DeepInfra API.

    Uses direct HTTP calls to the OpenAI-compatible API.
    Requires DEEPINFRA_API_KEY environment variable or config.
    """

    def __init__(self, config: DeepInfraConfig | None = None):
        self.config = config or DeepInfraConfig()
        self._api_key: str | None = None

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def dimension(self) -> int:
        return 1024

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self._api_key is None:
            import os
            self._api_key = self.config.api_key or os.getenv("DEEPINFRA_API_KEY")
            if not self._api_key:
                raise ValueError(
                    "DEEPINFRA_API_KEY not set. "
                    "Set it as environment variable or pass in DeepInfraConfig."
                )
            logger.info(f"DeepInfra API configured for {self.config.model}")
        return self._api_key

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using DeepInfra API with direct HTTP calls."""
        import urllib.request
        import json as json_module

        if not texts:
            return np.array([])

        api_key = self._get_api_key()
        embeddings = []

        logger.debug(f"Embedding {len(texts)} texts via DeepInfra API")

        # Process in batches to respect API limits
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            # Prepare request
            url = f"{self.config.base_url}/embeddings"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.config.model,
                "input": batch,
            }

            # Make HTTP request
            req = urllib.request.Request(
                url,
                data=json_module.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json_module.loads(response.read().decode("utf-8"))

            # Extract embeddings from response
            batch_embeddings = [item["embedding"] for item in result["data"]]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)


def create_embedding_strategy(
    strategy: str = "local",
    **kwargs,
) -> EmbeddingStrategy:
    """
    Factory function to create an embedding strategy.

    Args:
        strategy: Strategy name - "local" or "deepinfra"
        **kwargs: Additional config options passed to the config dataclass

    Returns:
        An EmbeddingStrategy instance

    Examples:
        # Local BGE-M3
        strategy = create_embedding_strategy("local")

        # DeepInfra with custom batch size
        strategy = create_embedding_strategy("deepinfra", batch_size=64)

        # Local with fp32
        strategy = create_embedding_strategy("local", use_fp16=False)
    """
    if strategy == "local":
        config = LocalBGEM3Config(**kwargs)
        return LocalBGEM3Embedding(config)
    elif strategy == "deepinfra":
        config = DeepInfraConfig(**kwargs)
        return DeepInfraBGEM3Embedding(config)
    else:
        raise ValueError(
            f"Unknown embedding strategy: {strategy}. "
            f"Available: 'local', 'deepinfra'"
        )
