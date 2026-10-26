"""Readable NumPy implementations of three classification algorithms."""

from __future__ import annotations

import numpy as np


class DecisionTreeClassifierScratch:
    """Binary decision tree using Gini impurity and greedy splits."""

    def __init__(self, max_depth=6, min_samples_split=20, max_thresholds=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.tree_ = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = np.mean(y)
        return 1.0 - p**2 - (1.0 - p) ** 2

    def _best_split(self, X, y):
        parent_impurity = self._gini(y)
        best_gain, best_feature, best_threshold = 0.0, None, None
        n_samples = len(y)

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            if len(values) < 2:
                continue
            if len(values) > self.max_thresholds + 1:
                quantiles = np.linspace(0, 100, self.max_thresholds + 2)[1:-1]
                thresholds = np.unique(np.percentile(values, quantiles))
            else:
                thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left = X[:, feature] <= threshold
                n_left = int(left.sum())
                if n_left == 0 or n_left == n_samples:
                    continue
                gain = parent_impurity - (
                    n_left / n_samples * self._gini(y[left])
                    + (n_samples - n_left) / n_samples * self._gini(y[~left])
                )
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        return best_feature, best_threshold

    def _grow(self, X, y, depth):
        prediction = int(np.mean(y) >= 0.5)
        node = {"prediction": prediction}
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return node
        left = X[:, feature] <= threshold
        node.update(
            feature=feature,
            threshold=float(threshold),
            left=self._grow(X[left], y[left], depth + 1),
            right=self._grow(X[~left], y[~left], depth + 1),
        )
        return node

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)
        self.tree_ = self._grow(X, y, 0)
        return self

    def predict(self, X):
        predictions = []
        for row in np.asarray(X):
            node = self.tree_
            while "feature" in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            predictions.append(node["prediction"])
        return np.asarray(predictions, dtype=np.int8)


class KNNClassifierScratch:
    """K-nearest neighbors with Euclidean distance and majority voting."""

    def __init__(self, n_neighbors=9, batch_size=128):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y, dtype=np.int8)
        self.train_norm_ = np.sum(self.X_train_**2, axis=1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        output = []
        for start in range(0, len(X), self.batch_size):
            batch = X[start : start + self.batch_size]
            distances = (
                np.sum(batch**2, axis=1)[:, None]
                + self.train_norm_[None, :]
                - 2.0 * batch @ self.X_train_.T
            )
            neighbors = np.argpartition(distances, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
            output.extend((self.y_train_[neighbors].mean(axis=1) >= 0.5).astype(np.int8))
        return np.asarray(output)


class PerceptronClassifierScratch:
    """Binary linear perceptron trained with shuffled online updates."""

    def __init__(self, learning_rate=0.01, epochs=30, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)
        self.weights_ = np.zeros(X.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            for index in rng.permutation(len(X)):
                prediction = int(X[index] @ self.weights_ + self.bias_ >= 0)
                update = self.learning_rate * (y[index] - prediction)
                self.weights_ += update * X[index]
                self.bias_ += update
        return self

    def predict(self, X):
        scores = np.asarray(X, dtype=np.float32) @ self.weights_ + self.bias_
        return (scores >= 0).astype(np.int8)
