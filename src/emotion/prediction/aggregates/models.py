# from catboost import CatBoostRegressor
from sklearn import linear_model
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Define the machine learning models to try
MODELS = [
    {
        "name": "MLPRegressor",
        "model": MLPRegressor(max_iter=1000),
        "params": {
            "hidden_layer_sizes": [(64, 32), (128, 64)],
            "alpha": [0.001, 0.01, 0.1],
            "learning_rate_init": [0.001, 0.01, 0.1],
        },
    },
    {
        "name": "RandomForestRegressor",
        "model": RandomForestRegressor(),
        "params": {"n_estimators": [50, 100, 200], "max_depth": range(2, 10)},
    },
    {
        "name": "ExtraTreesRegressor",
        "model": ExtraTreesRegressor(),
        "params": {"n_estimators": [50, 100, 200], "max_depth": range(2, 10)},
    },
    {
        "name": "GradientBoostingRegressor",
        "model": GradientBoostingRegressor(),
        "params": {
            "estimator__n_estimators": [50, 100, 200],
            "estimator__max_depth": range(2, 10),
            "estimator__learning_rate": [0.001, 0.01, 0.1],
        },
    },
    {
        "name": "SVR",
        "model": SVR(),
        "params": {
            "estimator__kernel": ["linear", "rbf"],
            "estimator__C": [0.1, 1.0, 10.0],
            "estimator__epsilon": [0.01, 0.1, 1.0],
        },
    },
    {
        "name": "AdaBoostRegressor",
        "model": AdaBoostRegressor(),
        "params": {
            "estimator__n_estimators": [50, 100, 200],
            "estimator__learning_rate": [0.001, 0.01, 0.1],
        },
    },
    {
        "name": "LinearRegression",
        "model": linear_model.LinearRegression(),
        "params": {},
    },
    {
        "name": "Ridge",
        "model": linear_model.Ridge(),
        "params": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    },
    {
        "name": "Lasso",
        "model": linear_model.Lasso(),
        "params": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    },
    # {
    #     "name": "CatBoostRegressor",
    #     "model": CatBoostRegressor(),
    #     "params": {"iterations": [10, 50, 100], "depth": [2, 4, 6]},
    #     "learning_rate": [0.001, 0.01, 0.1, 1.0],
    # },
]
