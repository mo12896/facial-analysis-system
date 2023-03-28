from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

MODELS = [
    {
        "name": "MLPRegressor",
        "model": MLPRegressor(max_iter=2000),
        "params": {
            "hidden_layer_sizes": [(64, 32), (128, 64)],
            "alpha": [0.001, 0.01, 0.1],
            "learning_rate_init": [0.001, 0.01, 0.1],
        },
    },
    {
        "name": "KNeighborsRegressor",
        "model": KNeighborsRegressor(),
        "params": {"n_neighbors": [1, 5, 10, 15, 20]},
    },
    {
        "name": "DecisionTreeRegressor",
        "model": DecisionTreeRegressor(),
        "params": {"max_depth": range(2, 11)},
    },
    {
        "name": "RandomForestRegressor",
        "model": RandomForestRegressor(),
        "params": {"n_estimators": [50, 100, 200, 400], "max_depth": range(2, 6)},
    },
    {
        "name": "ExtraTreesRegressor",
        "model": ExtraTreesRegressor(),
        "params": {"n_estimators": [50, 100, 200, 400], "max_depth": range(2, 6)},
    },
    {
        "name": "GradientBoostingRegressor",
        "model": GradientBoostingRegressor(),
        "params": {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": range(2, 6),
            "learning_rate": [0.001, 0.01, 0.1],
        },
    },
    {
        "name": "AdaBoostRegressor",
        "model": AdaBoostRegressor(),
        "params": {
            "n_estimators": [50, 100, 200, 400],
            "learning_rate": [0.001, 0.01, 0.1],
        },
    },
    {
        "name": "SVR",
        "model": SVR(),
        "params": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1.0, 10.0],
            "epsilon": [0.01, 0.1, 1.0],
        },
    },
    {
        "name": "LinearRegression",
        "model": LinearRegression(),
        "params": {},
    },
    {
        "name": "Ridge",
        "model": Ridge(),
        "params": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    },
    {
        "name": "Lasso",
        "model": Lasso(),
        "params": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    },
    {
        "name": "ElasticNet",
        "model": ElasticNet(),
        "params": {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
    },
    {
        "name": "BayesianRidge",
        "model": BayesianRidge(),
        "params": {
            "alpha_1": [0.001, 0.01, 0.1, 1.0],
            "alpha_2": [0.001, 0.01, 0.1, 1.0],
        },
    },
    {
        "name": "CatBoostRegressor",
        "model": CatBoostRegressor(verbose=False),
        "params": {
            "iterations": [50, 100, 200, 400],
            "depth": range(2, 6),
            "learning_rate": [0.001, 0.01, 0.1, 1.0],
        },
    },
    {
        "name": "XGBRegressor",
        "model": XGBRegressor(),
        "params": {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": range(2, 6),
            "learning_rate": [0.001, 0.01, 0.1],
        },
    },
]
