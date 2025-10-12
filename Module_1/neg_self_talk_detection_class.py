class SBERTClassifier:
    def __init__(self, embedder, classifier):
        self.embedder = embedder  # SentenceTransformer model
        self.classifier = classifier  # trained sklearn classifier

    def predict(self, texts):
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        return self.classifier.predict(embeddings)

    def decision_function(self, texts):
        if hasattr(self.classifier, "decision_function"):
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            return self.classifier.decision_function(embeddings)
        else:
            raise AttributeError("Classifier does not support decision_function")
