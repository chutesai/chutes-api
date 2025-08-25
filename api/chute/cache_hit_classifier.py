"""
Simple k-means cluster that uses ctps (combined tokens per second),
input token count, output token count, and compute multiplier
to predict if a prompt had some level of cache hit, thereby
decreasing latency and compute costs.
"""

import json
import numpy as np
from typing import Dict, Any


class CacheHitDetector:
    def __init__(self, params_path="/app/cache_hit_cluster_params.json"):
        with open(params_path, "r") as f:
            params = json.load(f)
        self.scaler_mean = np.array(params["scaler_mean"])
        self.scaler_scale = np.array(params["scaler_scale"])
        self.cluster_centers = np.array(params["cluster_centers"])
        self.cached_cluster = params["cached_cluster"]

    def predict(self, metrics: Dict[str, Any], compute_multiplier: float) -> bool:
        """
        Predict if a request was a cache hit
        """
        if not all(k in metrics for k in ["ctps", "it", "ot"]):
            return False
        if any(metrics[k] is None for k in ["ctps", "it", "ot"]):
            return False
        if metrics["ot"] < 10:
            return False
        X = np.array([compute_multiplier, metrics["it"], metrics["ot"], metrics["ctps"]])
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        distances = np.sum((self.cluster_centers - X_scaled) ** 2, axis=1)
        predicted_cluster = int(np.argmin(distances))
        return predicted_cluster == self.cached_cluster
