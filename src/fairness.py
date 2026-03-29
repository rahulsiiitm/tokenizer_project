# src/fairness.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

class FairnessAuditor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Custom vectorizer using our BPE tokenizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: self.tokenizer.encode(x)['tokens'], 
            token_pattern=None, # type: ignore
            lowercase=False
        )
        self.model = LogisticRegression()

    def train_sentiment_model(self, texts, labels):
        """Trains a downstream classifier to measure performance gaps[cite: 21]."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def get_token_saliency(self, text):
        """Maps token-level contributions back to original sub-words[cite: 38]."""
        tokens = self.tokenizer.encode(text)['tokens']
        if not tokens: return []
        
        feature_names = self.vectorizer.get_feature_names_out().tolist()
        coefs = self.model.coef_[0]
        
        # Determine weight for each token based on trained model coefficients
        saliency = []
        for token in tokens:
            if token in feature_names:
                idx = feature_names.index(token)
                saliency.append(coefs[idx])
            else:
                saliency.append(0.0)
        return saliency