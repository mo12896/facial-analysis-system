from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from catboost import Pool
from joblib import Parallel, delayed
from models import MODELS
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor


class HyperparaSearch:
    def __init__(
        self,
        models: List[Dict],
        n_folds: int = 5,
        metrics: List = ["mean_squared_error", "mean_absolute_error"],
    ):
        self.models = models
        self._n_folds = n_folds
        self._metrics = metrics

    # TODO: Implement random search, refactor the class + make CatBoost work
    # Define a helper function for parallelizing hyperparameter search
    def search_model(self, model_dict, X_train, y_train):
        results = []
        model_name = model_dict["name"]
        model = model_dict["model"]
        params = model_dict["params"]
        print(f"Model: {model_name}")

        kf = KFold(n_splits=self._n_folds, shuffle=True)

        for metric_name in self._metrics:
            if model_name == "CatBoostRegressor":
                train_pool = Pool(X_train, y_train)
                random_search = GridSearchCV(
                    model, params, cv=kf, scoring=f"neg_{metric_name}"
                )
                random_search.fit(train_pool, silent=True)
            elif model_name in [
                "GradientBoostingRegressor",
                "SVR",
                "AdaBoostRegressor",
            ]:
                multi_model = MultiOutputRegressor(model, n_jobs=-1)
                random_search = GridSearchCV(
                    multi_model,
                    params,
                    cv=kf,
                    scoring=f"neg_{metric_name}",
                )
                random_search.fit(X_train, y_train)
            else:
                random_search = GridSearchCV(
                    model, params, cv=kf, scoring=f"neg_{metric_name}"
                )
                random_search.fit(X_train, y_train)

            result = {
                "model": model_name,
                "params": random_search.best_params_,
                "metric": metric_name,
                "score": -random_search.best_score_,
            }
            results.append(result)
            print(f"{metric_name}: {-random_search.best_score_:.3f}")

        return results

    def run(self, X, y):
        results = Parallel(n_jobs=-1)(
            delayed(self.search_model)(model_dict, X, y) for model_dict in self.models
        )
        return results


if __name__ == "__main__":

    eval_metric = "mean_absolute_error"

    # Define the input and target vectors
    X = np.random.rand(100, 20)
    y = np.random.rand(100, 5)

    hyper_search = HyperparaSearch(models=MODELS)

    results = hyper_search.run(X, y)

    # Plot the results
    mae_scores = [
        rd["score"]
        for result_list in results
        for rd in result_list
        if rd["metric"] == eval_metric
    ]
    model_names = [
        rd["model"]
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
    print(f"Best model: {best_model['model']}")
    print(f"Best params: {best_model['params']}")
    print(f"Best Score: {best_model['score']}")
