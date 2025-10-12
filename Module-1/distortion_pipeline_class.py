import numpy as np
import re

class DistortionPipeline:
    def __init__(self, bin_model, mult_model, bin_vectorizer, mult_vectorizer,
                 bin_svd=None, mult_svd=None, bin_threshold=0.44):
        self.bin_model = bin_model
        self.mult_model = mult_model
        self.bin_vectorizer = bin_vectorizer
        self.mult_vectorizer = mult_vectorizer
        self.bin_svd = bin_svd
        self.mult_svd = mult_svd
        self.bin_threshold = bin_threshold

    # ---------- Preprocess inside the class ----------
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def predict(self, texts):
        # Preprocess
        texts_proc = [self.preprocess(t) for t in texts]

        # ---------- Binary prediction ----------
        X_bin_vec = self.bin_vectorizer.transform(texts_proc)
        if self.bin_svd:
            X_bin_vec = self.bin_svd.transform(X_bin_vec)

        bin_probs = self.bin_model.predict_proba(X_bin_vec)[:, 1]
        bin_preds = ['Distortion' if p >= self.bin_threshold else 'No Distortion'
                     for p in bin_probs]

        # ---------- Multiclass prediction for Distortions ----------
        final_preds = []
        for text, bin_pred in zip(texts_proc, bin_preds):
            if bin_pred == 'No Distortion':
                final_preds.append('No Distortion')
            else:
                # Vectorize for multiclass
                X_mult_vec = self.mult_vectorizer.transform([text])
                if self.mult_svd:
                    X_mult_vec = self.mult_svd.transform(X_mult_vec)
                mult_pred = self.mult_model.predict(X_mult_vec)[0]
                final_preds.append(mult_pred)

        return final_preds, bin_probs
