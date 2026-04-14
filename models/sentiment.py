"""
FinBERT sentiment scorer.

Model: ProsusAI/finbert  (fine-tuned on financial text)
Labels: positive, negative, neutral
Output: sentiment_score in [-1.0, +1.0]
    +1.0 = fully positive
     0.0 = neutral
    -1.0 = fully negative

First run downloads the model (~440 MB) from HuggingFace and caches it locally.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.logger import get_logger

logger = get_logger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
MAX_LENGTH = 512
BATCH_SIZE = 8


@dataclass
class SentimentResult:
    text: str
    label: str          # "positive" | "negative" | "neutral"
    score: float        # raw confidence (0-1) for the predicted label
    sentiment: float    # normalised [-1, +1]


@lru_cache(maxsize=1)
def _load_model():
    """Load FinBERT once and cache in memory."""
    logger.info(f"Loading FinBERT from '{FINBERT_MODEL}' (downloading on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    logger.info(f"FinBERT loaded on {device.upper()}")
    return tokenizer, model, device


class SentimentAnalyzer:
    """
    Score financial text using FinBERT.
    Thread-safe for sequential calls; not designed for concurrent use.
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is None:
            self._tokenizer, self._model, self._device = _load_model()

    def score(self, texts: list[str]) -> list[SentimentResult]:
        """
        Score a list of text strings.
        Returns one SentimentResult per input text.
        Empty strings return neutral (0.0).
        """
        if not texts:
            return []

        self._ensure_loaded()
        results = []

        # Process in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            # Replace empty strings
            batch = [t if t.strip() else "no news" for t in batch]

            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu()

            # FinBERT label order: positive=0, negative=1, neutral=2
            label_map = {0: "positive", 1: "negative", 2: "neutral"}

            for j, prob in enumerate(probs):
                idx = int(prob.argmax())
                label = label_map[idx]
                confidence = float(prob[idx])

                # Compute signed score: positive prob - negative prob
                sentiment = float(prob[0] - prob[1])

                results.append(SentimentResult(
                    text=batch[j],
                    label=label,
                    score=confidence,
                    sentiment=round(sentiment, 4),
                ))

        return results

    def score_symbol(self, texts: list[str]) -> float:
        """
        Aggregate sentiment across multiple headlines/articles for one symbol.
        Returns a single score in [-1.0, +1.0].
        Weights recent articles equally (simple mean).
        """
        if not texts:
            return 0.0
        results = self.score(texts)
        if not results:
            return 0.0
        avg = sum(r.sentiment for r in results) / len(results)
        return round(avg, 4)
