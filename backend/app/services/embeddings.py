import hashlib
import math
import random
from collections.abc import Sequence

import httpx

from app.core.config import settings


class EmbeddingService:
    """Service to generate text embeddings via OpenAI or offline deterministic fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION

    def _generate_deterministic_vector(self, text: str) -> list[float]:
        """Generate a deterministic, normalized vector for offline development and testing.

        Text with overlapping words will share vector components, enabling realistic similarity testing.
        """
        dim = self.dimension
        vec = [0.0] * dim
        words = text.lower().strip().split()
        if not words:
            words = ["empty"]

        for word in words:
            # Hash each word to seed a pseudo-random contribution
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            rng = random.Random(h)
            for _i in range(min(50, dim)):
                idx = rng.randint(0, dim - 1)
                vec[idx] += rng.uniform(-1.0, 1.0)

        # L2-normalize vector
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0

        return vec

    def get_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for a single string."""
        return self.get_embeddings([text])[0]

    def get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of strings."""
        if not texts:
            return []

        # If API key is present, call OpenAI Embeddings API
        if self.api_key:
            try:
                response = httpx.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": list(texts),
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                # Sort by index to maintain ordering
                sorted_items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_items]
            except Exception:
                # Fallback to deterministic vectors on network/API failure
                return [self._generate_deterministic_vector(t) for t in texts]

        # Offline / deterministic fallback
        return [self._generate_deterministic_vector(t) for t in texts]


embedding_service = EmbeddingService()
