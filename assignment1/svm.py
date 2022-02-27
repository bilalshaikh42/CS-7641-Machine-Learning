from load_data import PenDigitsDataset, SpamBaseDataset
from plots import makeAndPlotLearningCurve, plotConfusionMatrix, plotValidationCurve
from sklearn import svm
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def experiment(pipe, dataset):

    for kernel in ['linear', "poly", "rbf"]:
        break
        best_estimator = pipe
        best_estimator.set_params(svm__kernel=kernel)
        train_sizes = np.linspace(0.1, 1, 40, endpoint=True)

        makeAndPlotLearningCurve(best_estimator, f'SVM ({kernel} kernel)', dataset.xtrain, dataset.ytrain,
                                 train_sizes, dataset.scorer, dataset.score_label, "SVM", dataset.name, f'{kernel}-Base')

        cs = np.arange(0.0001, .02, 0.0001)

        plotValidationCurve(best_estimator, f'SVM ({kernel} kernel)', dataset.xtrain,
                            dataset.ytrain, "svm__C", cs, dataset.scorer, None, dataset.score_label, "SVM", dataset.name, f'{kernel}-c')

    best_estimator = pipe
    pipe.fit(dataset.xtrain, dataset.ytrain)
    plotValidationCurve(best_estimator, f'SVM ({kernel} kernel)', dataset.xtrain,
                        dataset.ytrain, "svm__kernel", ['linear', "poly", "rbf", "sigmoid"], dataset.scorer, None, dataset.score_label, "SVM", dataset.name, f'kernels')

    plotConfusionMatrix(best_estimator, dataset.xtest,
                        dataset.ytest, dataset.classes, "SVM", dataset.name)


def main():
    penDataset = PenDigitsDataset()
    dataset = SpamBaseDataset()
    svmStep = svm.SVC(kernel='linear')
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', svmStep)],
    )
    # pipe.set_params(C=1.0, kernel='linear')
    # pipe.fit(dataset.xtrain, dataset.ytrain)
    experiment(pipe, penDataset)
    experiment(pipe, dataset)


if __name__ == "__main__":
    main()
