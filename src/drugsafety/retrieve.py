"""Find the most similar labeled training sentences, used as worked examples (RAG few-shot)."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import Example


class ExampleRetriever:
    def __init__(self, train: list[Example]):
        self.train = train
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")
        self.mat = normalize(self.vec.fit_transform([e.text for e in train]))

    def top_k(self, text: str, k: int = 4, min_negatives: int = 1) -> list[Example]:
        """k nearest sentences, keeping at least `min_negatives` with no adverse event,
        so the model also sees what a "[]" answer looks like."""
        q = normalize(self.vec.transform([text]))
        sims = (self.mat @ q.T).toarray().ravel()
        order = sims.argsort()[::-1]
        picked = [self.train[i] for i in order[:k] if self.train[i].text != text]
        if sum(not e.has_ade for e in picked) < min_negatives:
            for i in order[k:]:
                e = self.train[i]
                if not e.has_ade and e.text != text:
                    picked = picked[: k - 1] + [e]
                    break
        return picked[:k]
