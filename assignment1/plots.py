from codecs import ignore_errors
from os import makedirs, mkdir
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms
from sklearn.metrics import ConfusionMatrixDisplay

# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


def makeAndPlotLearningCurve(estimator, title, x, y, train_sizes, scoring, metric_name, learner_name, dataset_name, condition_name):
    train_sizes, train_scores, test_scores, fit_times, predict_time = ms.learning_curve(
        estimator,
        x, y, cv=5, n_jobs=8,
        scoring=scoring,
        verbose=10,
        train_sizes=train_sizes,
        return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    train_points = train_scores_mean
    test_points = test_scores_mean
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    predict_times_mean = np.mean(predict_time, axis=1)
    predict_times_std = np.std(predict_time, axis=1)
    fig, ax = plt.subplots()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2)
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2)

    ax.plot(train_sizes, train_points, 'o-', linewidth=1, markersize=4,
            label="Training score")
    ax.plot(train_sizes, test_points, 'o-', linewidth=1, markersize=4,
            label="Cross-validation score")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Training set size")
    ax.title.set_text(f'Performance of {learner_name} ({dataset_name})')
    ax.legend(loc="best")
    makedirs("{}/{}".format(learner_name, dataset_name), exist_ok=True,)
    fig.savefig("{}/{}/{}learningCurve.png".format(learner_name,
                                                   dataset_name, condition_name))

    fig, ax = plt.subplots()

    ax.plot(train_sizes, fit_times_mean, "o-", label="Fit Time")
    ax.plot(train_sizes, predict_times_mean, "o-", label="Predict Time")
    ax.fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
        label="Fit Time"
    )
    ax.fill_between(
        train_sizes,
        predict_times_mean - predict_times_std,
        predict_times_mean + predict_times_std,
        alpha=0.1,
        label="Predict Time"
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Time (s)")
    ax.set_title(f'Model Time Performance {learner_name} ({dataset_name})')
    ax.legend(loc="best")
    makedirs("{}/{}".format(learner_name, dataset_name), exist_ok=True,)
    fig.tight_layout()
    fig.savefig("{}/{}/{}timingCurve.png".format(learner_name,
                                                 dataset_name, condition_name))


def plotValidationCurve(estimator, title, x, y, param_name, param_range, scoring, fit_params,  metric_name, learner_name, dataset_name, condition_name):
    train_scores, test_scores = ms.validation_curve(
        estimator,
        x, y, cv=5, n_jobs=8,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        fit_params=fit_params,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.clf()
    plt.title(f'Validation Curve {learner_name} ({dataset_name})')
    plt.xlabel(param_name.split("__")[1])
    plt.ylabel(metric_name)

    plt.plot(param_range, train_scores_mean, "o-", label="Training score")
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2
    )
    plt.plot(
        param_range, test_scores_mean, "o-", label="Cross-validation score",
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2
    )
    plt.legend(loc="best")
    makedirs("{}/{}".format(learner_name, dataset_name), exist_ok=True,)
    plt.savefig("{}/{}/{}validationCurve.png".format(learner_name,
                dataset_name, condition_name))


def plotPerformance(x_values, y_test, y_train, x_name, y_name, learner_name, dataset_name, condition_name, title="Model Performance", performance_metric="Accuracy"):
    fig, ax = plt.subplots()

    ax.plot(x_values, y_test, 'o-',  label='Test {}'.format(performance_metric))
    ax.plot(x_values, y_train, 'o-',
            label='Training {}'.format(performance_metric))

    ax.set_ylabel(y_name)
    ax.set_xlabel(x_name)

    ax.set_title(title)
    ax.legend(loc='best')
    fig.tight_layout()
    makedirs("{}/{}".format(learner_name, dataset_name), exist_ok=True,)
    fig.savefig("{}/{}/{}Performance.png".format(learner_name,
                dataset_name, condition_name))


def plotConfusionMatrix(classifier, X_test, y_test, class_names, learner_name, dataset_name, condition_name="final"):
    np.set_printoptions(formatter={'float': lambda x: format(x, '2f')})
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize="true",
    )

    disp.ax_.set_title(f'Confusion matrix {learner_name} ({dataset_name})')
    makedirs("{}/{}".format(learner_name, dataset_name), exist_ok=True,)
    plt.savefig("{}/{}/{}ConfusionMatrix.png".format(learner_name,
                dataset_name, condition_name))
