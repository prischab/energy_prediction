"""Feature selection utilities."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE


def run_rfe(X, y, n_features: int = 30, step: int = 5):
    """Run recursive feature elimination using a random forest regressor."""
    base = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rfe = RFE(base, n_features_to_select=n_features, step=step)
    rfe.fit(X, y)
    support = rfe.get_support()
    selected = X.columns[support].tolist()
    ranking = {col: int(rank) for col, rank in zip(X.columns, rfe.ranking_)}
    return selected, ranking
