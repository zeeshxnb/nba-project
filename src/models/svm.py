"""Support Vector Machine model for playoff series prediction."""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np


class SVMModel:
    """SVM classifier for predicting playoff series winners."""
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.scaler = StandardScaler()
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,  # Enable predict_proba
            random_state=42
        )
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X, y, feature_names=None):
        """Train the model on features X and labels y."""
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X):
        """Return binary predictions (1 = team_a wins)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Return probability that team_a wins."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(self, X, y, cv=5):
        """Return cross-validated accuracy."""
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        return {
            'mean_accuracy': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
