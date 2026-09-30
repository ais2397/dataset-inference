import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind


class DatasetInferenceDefense:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.regressor = None

    def blind_walk(self, model, x, y, max_steps=10, step_size=0.01, trials=10):
        """
        Performs Blind Walk to estimate prediction margin for a given data point.
        """
        x = x.unsqueeze(0).to(self.device)
        y = y.item()
        model.eval()
        model = model.to(self.device)  # Ensure model is on correct device
        distances = []

        with torch.no_grad():
            for _ in range(trials):
                direction = torch.randn_like(x).sign() * step_size
                for i in range(1, max_steps + 1):
                    x_perturbed = x + i * direction
                    pred = model(x_perturbed).argmax(dim=1).item()
                    if pred != y:
                        distances.append(i * step_size)
                        break
                else:
                    distances.append(max_steps * step_size)

        return np.array(distances)

    def generate_embeddings(self, model, dataset, num_points=50, **kwargs):
        """
        Generates embeddings using Blind Walk for all points in a dataset.
        """
        embeddings = []
        labels = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for i, (x, y) in enumerate(loader):
            if i >= num_points:
                break
            emb = self.blind_walk(model, x[0], y[0], **kwargs)
            embeddings.append(emb)
            labels.append(y.item())

        return np.array(embeddings), np.array(labels)

    def train_regressor(self, victim_model, private_dataset, public_dataset, num_points=50, **kwargs):
        """
        Trains logistic regression classifier to distinguish private vs public embeddings.
        """
        priv_emb, _ = self.generate_embeddings(victim_model, private_dataset, num_points=num_points, **kwargs)
        pub_emb, _ = self.generate_embeddings(victim_model, public_dataset, num_points=num_points, **kwargs)

        X = np.vstack([priv_emb, pub_emb])
        y = np.array([1] * len(priv_emb) + [0] * len(pub_emb))

        self.regressor = LogisticRegression(max_iter=1000)
        self.regressor.fit(X, y)
        return self.regressor

    def run_inference(self, suspect_model, private_dataset, public_dataset, num_points=50, alpha=0.01, **kwargs):
        """
        Applies trained regressor to embeddings from suspect model and runs statistical test.
        """
        priv_emb, _ = self.generate_embeddings(suspect_model, private_dataset, num_points=num_points, **kwargs)
        pub_emb, _ = self.generate_embeddings(suspect_model, public_dataset, num_points=num_points, **kwargs)

        priv_scores = self.regressor.predict_proba(priv_emb)[:, 1]
        pub_scores = self.regressor.predict_proba(pub_emb)[:, 1]

        # Hypothesis test: H0: µ_priv ≤ µ_pub
        t_stat, p_value = ttest_ind(priv_scores, pub_scores, alternative='greater')

        result = "STOLEN" if p_value < alpha else "INCONCLUSIVE"
        return {
            "mean_private_score": np.mean(priv_scores),
            "mean_public_score": np.mean(pub_scores),
            "p_value": p_value,
            "inference_result": result
        }
