from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from joblib import Parallel, delayed, dump
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier

from src.emotion.utils.constants import CUSTOM_MODEL_DIR


class Classifier(ABC):
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


class UnivariateClassifier(Classifier):
    def grid_search(self, X_train, y_train, kf, metric_name) -> GridSearchCV:
        random_search = GridSearchCV(
            self.model, self.params, cv=kf, scoring=metric_name
        )
        random_search.fit(X_train, y_train)
        return random_search

    def random_search(
        self, X_train, y_train, n_iter, kf, metric_name
    ) -> RandomizedSearchCV:
        random_search = RandomizedSearchCV(
            self.model, self.params, n_iter=n_iter, cv=kf, scoring=metric_name
        )
        random_search.fit(X_train, y_train)
        return random_search


class DefaultMultiVariateClassifier(Classifier):
    def grid_search(self, X_train, y_train, kf, metric_name) -> GridSearchCV:
        random_search = GridSearchCV(
            self.model, self.params, cv=kf, scoring=metric_name
        )
        random_search.fit(X_train, y_train)
        return random_search

    def random_search(
        self, X_train, y_train, n_iter, kf, metric_name
    ) -> RandomizedSearchCV:
        random_search = RandomizedSearchCV(
            self.model, self.params, n_iter=n_iter, cv=kf, scoring=metric_name
        )
        random_search.fit(X_train, y_train)
        return random_search


class CustomMultiVariateClassifier(Classifier):
    def grid_search(self, X_train, y_train, kf, metric_name) -> GridSearchCV:
        multi_model = MultiOutputClassifier(self.model, n_jobs=-1)
        random_search = GridSearchCV(
            multi_model, self.params, cv=kf, scoring=metric_name
        )
        random_search.fit(X_train, y_train)
        return random_search

    def random_search(
        self, X_train, y_train, n_iter, kf, metric_name
    ) -> RandomizedSearchCV:
        multi_model = MultiOutputClassifier(self.model, n_jobs=-1)
        random_search = RandomizedSearchCV(
            multi_model, self.params, n_iter=n_iter, cv=kf, scoring=metric_name
        )
        random_search.fit(X_train, y_train)
        return random_search


def create_classifier(model_name, model, params, mode: str = "multi") -> Classifier:
    if (
        model_name
        in [
            "GradientBoostingClassifier",
            "SVC",
            "AdaBoostClassifier",
            "CatBoostClassifier",
            "BayesianRidge",
        ]
        and mode == "multi"
    ):
        # Add estimator__ prefix to params
        params = {"estimator__" + key: value for key, value in params.items()}
        return CustomMultiVariateClassifier(model, params)
    elif mode == "multi":
        return DefaultMultiVariateClassifier(model, params)
    elif mode == "uni":
        return UnivariateClassifier(model, params)
    else:
        raise ValueError("Invalid mode")


class HyperparaSearchClassifier:
    def __init__(
        self,
        models: List[Dict],
        models_path: Path = CUSTOM_MODEL_DIR / "aggregates",
        n_folds: int = 5,
        metrics: List = ["accuracy"],
        n_jobs: int = -1,
        mode: str = "uni",
    ):
        self.models = models
        self.models_path = models_path
        self._n_folds = n_folds
        self._metrics = metrics
        self.n_jobs = n_jobs
        self.mode = mode

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
        # if self.verbose:
        #     print(f"Model: {model_name}")

        kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=42)

        for metric_name in self._metrics:
            regressor = create_classifier(model_name, model, params, mode=self.mode)

            result = regressor.grid_search(X_train, y_train, kf, metric_name)

            result_dict = {
                "name": model_name,
                "params": result.best_params_,
                "metric": metric_name,
                "score": result.best_score_,
                "model": result,
            }
            results.append(result_dict)
            # if self.verbose:
            #     print(f"{metric_name}: {-result.best_score_:.3f}")

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
