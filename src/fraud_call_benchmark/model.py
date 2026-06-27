from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class HashingVectorizer:
    """A lightweight character n-gram hashing vectorizer without sklearn."""

    def __init__(self, n_features: int = 4096, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self.n_features = n_features
        self.ngram_range = ngram_range

    def _iter_ngrams(self, text: str) -> Iterable[str]:
        clean = " ".join(text.split())
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if len(clean) < n:
                continue
            for idx in range(len(clean) - n + 1):
                yield clean[idx : idx + n]

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.n_features), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in self._iter_ngrams(text):
                col_idx = hash(token) % self.n_features
                matrix[row_idx, col_idx] += 1.0
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums


@dataclass
class LogisticConfig:
    learning_rate: float = 0.3
    epochs: int = 60
    l2: float = 1e-4
    batch_size: int = 64
    random_state: int = 42


class NumpyLogisticRegression:
    """Binary logistic regression trained with mini-batch gradient descent."""

    def __init__(self, config: LogisticConfig | None = None) -> None:
        self.config = config or LogisticConfig()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.config.random_state)
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0

        indices = np.arange(n_samples)
        for _ in range(self.config.epochs):
            rng.shuffle(indices)
            for start in range(0, n_samples, self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]
                xb = x[batch_idx]
                yb = y[batch_idx]

                logits = xb @ self.weights + self.bias
                preds = self._sigmoid(logits)
                errors = preds - yb

                grad_w = (xb.T @ errors) / len(batch_idx) + self.config.l2 * self.weights
                grad_b = float(errors.mean())

                self.weights -= self.config.learning_rate * grad_w
                self.bias -= self.config.learning_rate * grad_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("模型尚未训练。")
        logits = x @ self.weights + self.bias
        return self._sigmoid(logits)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(int)
