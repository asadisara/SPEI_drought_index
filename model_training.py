import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def build_pipelines():
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(random_state=42))
    ])

    return rf_pipeline, xgb_pipeline


def get_param_grids():
    rf_param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 20, 30, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }

    xgb_param_grid = {
        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1, 0.3],
        'xgb__subsample': [0.8, 0.9, 1.0]
    }

    return rf_param_grid, xgb_param_grid


def train_models(X_train, y_train, rf_pipeline, xgb_pipeline, rf_param_grid, xgb_param_grid):
    tscv = TimeSeriesSplit(n_splits=5)

    rf_random = RandomizedSearchCV(
        rf_pipeline, rf_param_grid, n_iter=20, cv=tscv,
        scoring='neg_mean_squared_error', n_jobs=-1, random_state=42,
        return_train_score=True
    )

    xgb_random = RandomizedSearchCV(
        xgb_pipeline, xgb_param_grid, n_iter=20, cv=tscv,
        scoring='neg_mean_squared_error', n_jobs=-1, random_state=42,
        return_train_score=True
    )

    rf_random.fit(X_train, y_train)
    xgb_random.fit(X_train, y_train)

    return rf_random, xgb_random
