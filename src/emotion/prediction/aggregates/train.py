# import os
# import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, dump
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor

from src.emotion.prediction.aggregates.models import MODELS
from src.emotion.utils.constants import CUSTOM_MODEL_DIR

# parent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(parent_folder)


class Regressor(ABC):
    def __init__(self, model, params):
        self.model = model
        self.params = params

    @abstractmethod
    def grid_search(self, X_train, y_train, kf, metric_name) -> GridSearchCV:
        """Grid search for hyperparameters

        Args:
            X_train (_type_): Independent variables
            y_train (_type_): Dependent variables
            kf (_type_): k-fold cross validation
            metric_name (_type_): Metric to optimize

        Returns:
            GridSearchCV: Grid Search object
        """

    @abstractmethod
    def random_search(
        self, X_train, y_train, n_iter, kf, metric_name
    ) -> RandomizedSearchCV:
        """Random search for hyperparameters

        Args:
            X_train (_type_): Independent variables
            y_train (_type_): Dependent variables
            n_iter (_type_): Number of iterations
            kf (_type_): k-fold cross validation
            metric_name (_type_): Metric to optimize

        Returns:
            GridSearchCV: Grid Search object
        """


class UnivariateRegressor(Regressor):
    def grid_search(self, X_train, y_train, kf, metric_name) -> GridSearchCV:
        if y_train.ndim == 1:
            # If y_train has only one output, fit a single model
            random_search = GridSearchCV(
                self.model, self.params, cv=kf, scoring=f"neg_{metric_name}"
            )
            random_search.fit(X_train, y_train)
        else:
            # Fit multiple models for each output
            random_search = GridSearchCV(
                MultiOutputRegressor(self.model),
                self.params,
                cv=kf,
                scoring=f"neg_{metric_name}",
            )
            random_search.fit(X_train, y_train)
        return random_search

    def random_search(
        self, X_train, y_train, n_iter, kf, metric_name
    ) -> RandomizedSearchCV:
        random_search = RandomizedSearchCV(
            self.model, self.params, n_iter=n_iter, cv=kf, scoring=f"neg_{metric_name}"
        )
        random_search.fit(X_train, y_train)
        return random_search


class DefaultMultiVariateRegressor(Regressor):
    def grid_search(self, X_train, y_train, kf, metric_name) -> GridSearchCV:
        random_search = GridSearchCV(
            self.model, self.params, cv=kf, scoring=f"neg_{metric_name}"
        )
        random_search.fit(X_train, y_train)
        return random_search

    def random_search(
        self, X_train, y_train, n_iter, kf, metric_name
    ) -> RandomizedSearchCV:
        random_search = RandomizedSearchCV(
            self.model, self.params, n_iter=n_iter, cv=kf, scoring=f"neg_{metric_name}"
        )
        random_search.fit(X_train, y_train)
        return random_search


class CustomMultiVariateRegressor(Regressor):
    def grid_search(self, X_train, y_train, kf, metric_name) -> GridSearchCV:
        multi_model = MultiOutputRegressor(self.model, n_jobs=-1)
        random_search = GridSearchCV(
            multi_model, self.params, cv=kf, scoring=f"neg_{metric_name}"
        )
        random_search.fit(X_train, y_train)
        return random_search

    def random_search(
        self, X_train, y_train, n_iter, kf, metric_name
    ) -> RandomizedSearchCV:
        multi_model = MultiOutputRegressor(self.model, n_jobs=-1)
        random_search = RandomizedSearchCV(
            multi_model, self.params, n_iter=n_iter, cv=kf, scoring=f"neg_{metric_name}"
        )
        random_search.fit(X_train, y_train)
        return random_search


def create_regressor(model_name, model, params, mode: str = "multi") -> Regressor:
    if (
        model_name
        in [
            "GradientBoostingRegressor",
            "SVR",
            "AdaBoostRegressor",
            "CatBoostRegressor",
            "BayesianRidge",
        ]
        and mode == "multi"
    ):
        # Add estimator__ prefix to params
        params = {"estimator__" + key: value for key, value in params.items()}
        return CustomMultiVariateRegressor(model, params)
    elif mode == "multi":
        return DefaultMultiVariateRegressor(model, params)
    elif mode == "uni":
        return UnivariateRegressor(model, params)
    else:
        raise ValueError("Invalid mode")


# TODO: Use random search instead of grid search!
class HyperparaSearch:
    def __init__(
        self,
        models: List[Dict],
        models_path: Path = CUSTOM_MODEL_DIR / "aggregates",
        n_folds: int = 5,
        metrics: List = ["mean_squared_error", "mean_absolute_error"],
        n_jobs: int = -1,
        mode: str = "multi",
        verbose: bool = False,
    ):
        self.models = models
        self.models_path = models_path
        self._n_folds = n_folds
        self._metrics = metrics
        self.n_jobs = n_jobs
        self.mode = mode
        self.verbose = verbose

    # Define a helper function for parallelizing hyperparameter search
    def _search_model(
        self,
        model_dict,
        X_train,
        y_train,
    ):
        results = []
        model_name = model_dict["name"]
        model = model_dict["model"]
        params = model_dict["params"]
        if self.verbose:
            print(f"Model: {model_name}")

        kf = KFold(n_splits=self._n_folds, shuffle=True)

        for metric_name in self._metrics:
            regressor = create_regressor(model_name, model, params, mode=self.mode)

            result = regressor.grid_search(X_train, y_train, kf, metric_name)

            # Compute the feature importances
            if self.mode == "multi":
                if model_name in ["CatBoostRegressor"]:
                    importances = []
                    for i in range(y_train.shape[1]):
                        importances.append(
                            result.best_estimator_.estimators_[
                                i
                            ].get_feature_importance()
                        )
                    feat_importance = np.mean(importances, axis=0)
                elif model_name in ["AdaBoostRegressor"]:
                    importances = []
                    for i in range(y_train.shape[1]):
                        importances.append(
                            result.best_estimator_.estimators_[i].feature_importances_
                        )
                    feat_importance = np.mean(importances, axis=0)
                elif model_name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
                    feat_importance = result.best_estimator_.coef_
                elif model_name in ["BayesianRidge"]:
                    importances = []
                    for i in range(y_train.shape[1]):
                        importances.append(result.best_estimator_.estimators_[i].coef_)
                    feat_importance = np.mean(importances, axis=0)
                elif model_name in ["SVR", "KNeighborsRegressor", "MLPRegressor"]:
                    feat_importance = None
                else:
                    feat_importance = result.best_estimator_.feature_importances_
            else:
                if model_name in ["CatBoostRegressor"]:
                    feat_importance = result.best_estimator_.get_feature_importance()
                elif model_name in [
                    "LinearRegression",
                    "Ridge",
                    "Lasso",
                    "ElasticNet",
                    "BayesianRidge",
                ]:
                    feat_importance = result.best_estimator_.coef_
                elif model_name in ["SVR", "KNeighborsRegressor", "MLPRegressor"]:
                    feat_importance = None
                else:
                    feat_importance = result.best_estimator_.feature_importances_

            result_dict = {
                "name": model_name,
                "params": result.best_params_,
                "metric": metric_name,
                "score": -result.best_score_,
                "model": result,
                "best_feats": feat_importance,
            }
            results.append(result_dict)
            if self.verbose:
                print(f"{metric_name}: {-result.best_score_:.3f}")

        return results

    def _save_models(self, search_results):
        self.models_path.mkdir(parents=True, exist_ok=True)

        for result in search_results:
            model_name = result[0]["name"]
            model_path = self.models_path / f"{model_name}.joblib"
            model = [metric_result["model"] for metric_result in result]
            dump(model, model_path)
            print(f"Models for {model_name} saved to {model_path}")

    def run(self, X, Y, save: bool = False):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._search_model)(model_dict, X, Y) for model_dict in self.models
        )
        if save:
            self._save_models(results)

        return results


if __name__ == "__main__":

    eval_metric = "mean_absolute_error"

    # Define the input and target vectors
    X = np.random.rand(100, 20)
    y = np.random.rand(100, 5)

    search = HyperparaSearch(models=MODELS)

    results = search.run(X, y, save=True)

    # Plot the results
    mae_scores = [
        rd["score"]
        for result_list in results
        for rd in result_list
        if rd["metric"] == eval_metric
    ]
    model_names = [
        rd["name"]
        for result_list in results
        for rd in result_list
        if rd["metric"] == eval_metric
    ]
    plt.bar(model_names, mae_scores)
    plt.title("Mean Absolute Error Scores")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.show()

    # Print the model with the lowest score
    best_model = min(
        [
            min(
                sublist,
                key=lambda x: x["score"]
                if x["metric"] == eval_metric
                else float("inf"),
            )
            for sublist in results
        ],
        key=lambda x: x["score"],
    )
    print(f"Best model: {best_model['name']}")
    print(f"Best params: {best_model['params']}")
    print(f"Best Score: {best_model['score']}")
