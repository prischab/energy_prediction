from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor


def run_rfe(X, y, n_features=30, step=5):
base = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rfe = RFE(base, n_features_to_select=n_features, step=step)
rfe.fit(X, y)
support = rfe.get_support()
selected = X.columns[support].tolist()
ranking = {c: int(r) for c, r in zip(X.columns, rfe.ranking_)}
return selected, ranking