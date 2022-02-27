import warnings
import sklearn
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from plots import plotConfusionMatrix, plotPerformance, makeAndPlotLearningCurve, plotValidationCurve
from load_data import PenDigitsDataset, SpamBaseDataset
from sklearn import tree, ensemble
from sklearn.model_selection import GridSearchCV as grid_search
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from multiprocessing import Pool
from load_data import PenDigitsDataset, SpamBaseDataset

from plots import makeAndPlotLearningCurve, plotPerformance


class Boost(object):
    def __init__(self) -> None:
        self.name = "AdaBoost"
        self.boost = ensemble.AdaBoostClassifier(
            algorithm="SAMME",
            n_estimators=100,
            learning_rate=1.0,

        )


def estimators(pipe, dataset):
    n_estimators = range(1, 200)
    algorithms = ["SAMME", "SAMME.R"]

    for algorithm in algorithms:
        score_train = []
        score_test = []
        prec_train = []
        prec_test = []
        for estimator in n_estimators:
            print("Running adaboost with estimator: ", estimator)
            pipe.set_params(boost__n_estimators=estimator)
            pipe.fit(dataset.xtrain, dataset.ytrain)
            y_train_pred = pipe.predict(dataset.xtrain)
            y_test_pred = pipe.predict(dataset.xtest)
            score_train.append(accuracy_score(dataset.ytrain, y_train_pred))
            score_test.append(accuracy_score(dataset.ytest, y_test_pred))

        plotPerformance(n_estimators, score_train, score_test, "Number of Estimators",
                        "Accuracy", "Adaboost", dataset.name, f'nEstimators-{algorithm}')


def experiments(pipe, dataset, params=None):
    train_sizes = np.linspace(0.1, 1, 40, endpoint=True)
    makeAndPlotLearningCurve(pipe, "Adaboost", dataset.xtrain, dataset.ytrain,
                             train_sizes, "accuracy", "Accuracy", "Adaboost", dataset.name, "DefaultBase")
    plotValidationCurve(pipe, "decisionTree", dataset.xtrain, dataset.ytrain, "boost__n_estimators", np.arange(
        1, 200, 1), "accuracy", None, "Accuracy", "Adaboost", dataset.name, "DefaultnEstimators")
    plotValidationCurve(pipe, "decisionTree", dataset.xtrain, dataset.ytrain, "boost__learning_rate",  [
                        0.1, 0.5, 1.0, 1.5, 2.0], "accuracy", None, "Accuracy", "Adaboost", dataset.name, "DefaultlearningRate")
    plotValidationCurve(pipe, "decisionTree", dataset.xtrain, dataset.ytrain, "boost__algorithm", [
                        "SAMME", "SAMME.R"], "accuracy", None, "Accuracy", "Adaboost", dataset.name, "Defaultalgorithm")

    if params is None:
        params = {
            "boost__n_estimators": range(1, 200),
            "boost__learning_rate": [0.1, 0.5, 1.0, 1.5, 2.0],
            "boost__algorithm": ["SAMME", "SAMME.R"]
        }
        grid_search = ms.GridSearchCV(
            pipe, n_jobs=8, param_grid=params, refit=True, verbose=10, cv=5, scoring="accuracy",
        )
        grid_search.fit(dataset.xtrain, dataset.ytrain,)

        best_estimator = grid_search.best_estimator_
        best_estimator.set_params()
        best_estimator.fit(
            dataset.xtrain, dataset.ytrain)
    else:
        best_estimator = pipe
        best_estimator.set_params(**params)
        best_estimator.fit(dataset.xtrain, dataset.ytrain)

    # makeAndPlotLearningCurve(best_estimator, "Adaboost", dataset.xtrain, dataset.ytrain,
    #                          train_sizes, "accuracy", "Accuracy", "Adaboost", dataset.name, "Base")

    # plotValidationCurve(best_estimator, "decisionTree", dataset.xtrain, dataset.ytrain, "boost__n_estimators", np.arange(
    #     1, 200, 1), "accuracy", None, "Accuracy", "Adaboost", dataset.name, "nEstimators")
    # plotValidationCurve(best_estimator, "decisionTree", dataset.xtrain, dataset.ytrain, "boost__learning_rate",  [
    #                     0.1, 0.5, 1.0, 1.5, 2.0], "accuracy", None, "Accuracy", "Adaboost", dataset.name, "learningRate")
    # plotValidationCurve(best_estimator, "decisionTree", dataset.xtrain, dataset.ytrain, "boost__algorithm", [
    #                     "SAMME", "SAMME.R"], "accuracy", None, "Accuracy", "Adaboost", dataset.name, "algorithm")
    plotConfusionMatrix(best_estimator, dataset.xtest,
                        dataset.ytest, dataset.classes, "Adaboost", dataset.name)


def main():
    warnings.simplefilter("ignore", sklearn.exceptions.DataConversionWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    pendataset = PenDigitsDataset()
    dataset = SpamBaseDataset()
    weak_learner = tree.DecisionTreeClassifier(
        max_depth=3,
    )
    boost = ensemble.AdaBoostClassifier(
        base_estimator=weak_learner,
        algorithm="SAMME",
        n_estimators=100,
        learning_rate=1.0,

    )

    pipe = Pipeline([("scaler", StandardScaler()), ('boost', boost)])
    experiments(pipe, dataset)
    experiments(pipe, pendataset)


if(__name__ == "__main__"):
    main()
