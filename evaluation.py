import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def nse(obs, sim):
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def pbias(obs, sim):
    return 100 * np.sum(sim - obs) / np.sum(obs)

def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "MSE": mean_squared_error(y, y_pred),
        "R2": r2_score(y, y_pred),
        "NSE": nse(y, y_pred),
        "PBIAS": pbias(y, y_pred),
        "KGE": kge(y, y_pred)
    }
