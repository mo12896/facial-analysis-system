from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

CLASSIFIER = [
    {
        "name": "MLPClassifier",
        "model": MLPClassifier(max_iter=2000),
        "params": {
            "hidden_layer_sizes": [(64, 32), (128, 64)],
            "alpha": [0.001, 0.01, 0.1],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "random_state": [42],
        },
    },
    {
        "name": "KNeighborsClassifier",
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [1, 5, 10, 15, 20]},
    },
    {
        "name": "DecisionTreeClassifier",
        "model": DecisionTreeClassifier(),
        "params": {"max_depth": range(2, 11)},
        "random_state": [42],
    },
    {
        "name": "RandomForestClassifier",
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": range(2, 6),
            "class_weight": ["balanced", "balanced_subsample"],
            "random_state": [42],
        },
    },
    {
        "name": "ExtraTreesClassifier",
        "model": ExtraTreesClassifier(),
        "params": {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": range(2, 6),
            "class_weight": ["balanced", "balanced_subsample"],
            "random_state": [42],
        },
    },
    {
        "name": "GradientBoostingClassifier",
        "model": GradientBoostingClassifier(),
        "params": {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": range(2, 6),
            "learning_rate": [0.001, 0.01, 0.1],
            "random_state": [42],
        },
    },
    {
        "name": "AdaBoostClassifier",
        "model": AdaBoostClassifier(),
        "params": {
            "n_estimators": [50, 100, 200, 400],
            "learning_rate": [0.001, 0.01, 0.1],
            "random_state": [42],
        },
    },
    {
        "name": "SVC",
        "model": SVC(),
        "params": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1.0, 10.0],
            "shrinking": [True, False],
            "class_weight": ["balanced"],
            "random_state": [42],
        },
    },
    # 160/300 fits fails, because the grid search is exhaustive!
    {
        "name": "LogisticRegression",
        "model": LogisticRegression(max_iter=2000),
        "params": {
            "C": [0.001, 0.01, 0.1, 1.0],
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "class_weight": ["balanced"],
            "random_state": [42],
        },
    },
    {
        "name": "RidgeClassifier",
        "model": RidgeClassifier(),
        "params": {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "class_weight": ["balanced"],
            "random_state": [42],
        },
    },
    {
        "name": "GaussianNB",
        "model": GaussianNB(),
        "params": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    },
    {
        "name": "CatBoostClassifier",
        "model": CatBoostClassifier(verbose=False),
        "params": {
            "iterations": [50, 100, 500, 1000],
            "depth": range(2, 6),
            "learning_rate": [0.001, 0.01, 0.1, 1.0],
            "auto_class_weights": ["Balanced", "SqrtBalanced"],
            "random_state": [42],
        },
    },
    {
        "name": "XGBClassifier",
        "model": XGBClassifier(),
        "params": {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": range(2, 6),
            "learning_rate": [0.001, 0.01, 0.1],
            "scale_pos_weight": [1, 50, 99, 1000],
            "random_state": [42],
        },
    },
]
