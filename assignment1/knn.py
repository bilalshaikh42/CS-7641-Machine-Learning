from sklearn import neighbors
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from plots import makeAndPlotLearningCurve, plotConfusionMatrix, plotPerformance, plotValidationCurve
from load_data import PenDigitsDataset, SpamBaseDataset


def distanceMetric(pipe, dataset):
    ps = range(1, 10)
    score_train = []
    score_test = []
    prec_train = []
    prec_test = []
    for p in ps:
        print("p: ", p)
        pipe.set_params(KNN__p=p, KNN__n_neighbors=2)
        pipe.fit(dataset.xtrain, dataset.ytrain)
        y_train_pred = pipe.predict(dataset.xtrain)
        y_test_pred = pipe.predict(dataset.xtest)
        score_train.append(accuracy_score(dataset.ytrain, y_train_pred))
        score_test.append(accuracy_score(dataset.ytest, y_test_pred))
        prec_train.append(precision_score(
            dataset.ytrain, y_train_pred, average='macro'))
        prec_test.append(precision_score(
            dataset.ytest, y_test_pred, average='macro'))

    plotPerformance(ps, score_train, score_test, "Value of P in Minkowski",
                    "Accuracy", "KNN", dataset.name, "KNN_p")


def neighborsComplexity(pipe, dataset):
    num_neighbors = range(1, 50)
    score_train = []
    score_test = []
    prec_train = []
    prec_test = []
    for num in num_neighbors:
        print("num neighbors: ", num)
        pipe.set_params(KNN__n_neighbors=num)
        pipe.fit(dataset.xtrain, dataset.ytrain)
        y_train_pred = pipe.predict(dataset.xtrain)
        y_test_pred = pipe.predict(dataset.xtest)
        score_train.append(accuracy_score(dataset.ytrain, y_train_pred))
        score_test.append(accuracy_score(dataset.ytest, y_test_pred))
        prec_train.append(precision_score(
            dataset.ytrain, y_train_pred, average='macro'))
        prec_test.append(precision_score(
            dataset.ytest, y_test_pred, average='macro'))
    plotPerformance(num_neighbors, score_train, score_test, "Number of neighbors",
                    "Accuracy", "KNN", dataset.name, "KNN_neighbors")


def experiment(pipe, dataset):
    num_neighbors = range(1, 50)
    train_sizes = np.linspace(0.1, 1, 40, endpoint=True)
    best_estimator = pipe

    makeAndPlotLearningCurve(best_estimator, "decisionTree", dataset.xtrain, dataset.ytrain,
                             train_sizes, "accuracy", "Accuracy", "KNN", dataset.name, "DefaultBase")
    makeAndPlotLearningCurve(best_estimator, "decisionTree", dataset.xtrain, dataset.ytrain,
                             train_sizes, "balanced_accuracy", "Accuracy", "KNN", dataset.name, "BaseBalanced")
    makeAndPlotLearningCurve(best_estimator, "decisionTree", dataset.xtrain, dataset.ytrain,
                             train_sizes, "f1_micro", "F1 Score", "KNN", dataset.name, "BaseF1")
    plotValidationCurve(best_estimator, "KNN", dataset.xtrain, dataset.ytrain, "KNN__n_neighbors",
                        num_neighbors, "accuracy", None, "Accuracy", "KNN", dataset.name, "KNN_neighbors")
    pipe.set_params(KNN__metric='minkowski', KNN__n_neighbors=4)
    plotValidationCurve(best_estimator, "KNN", dataset.xtrain, dataset.ytrain, "KNN__p", np.arange(
        1, 10), "accuracy", None, "Accuracy", "KNN", dataset.name, "KNN_p")
    plotValidationCurve(best_estimator, "KNN", dataset.xtrain, dataset.ytrain, "KNN__leaf_size", np.arange(
        1, 40), "accuracy", None, "Accuracy", "KNN", dataset.name, "KNN_leaf")
    best_estimator.fit(dataset.xtrain, dataset.ytrain)
    plotConfusionMatrix(best_estimator, dataset.xtest,
                        dataset.ytest, dataset.classes, "KNN", dataset.name)


def main():

    Pendataset = PenDigitsDataset()

    knn = neighbors.KNeighborsClassifier(n_neighbors=5,
                                         weights='uniform',
                                         algorithm='auto',
                                         leaf_size=30,
                                         p=2,
                                         metric='minkowski',
                                         metric_params=None,
                                         n_jobs=1)
    dataset = SpamBaseDataset()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("KNN", knn)])
    neighborsComplexity(pipe, dataset)
    neighborsComplexity(pipe, Pendataset)
    distanceMetric(pipe, Pendataset)
    distanceMetric(pipe, dataset)
    experiment(pipe, dataset)
    experiment(pipe, Pendataset)


if __name__ == "__main__":
    main()
