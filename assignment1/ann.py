from sklearn import neural_network
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from plots import makeAndPlotLearningCurve, plotConfusionMatrix, plotPerformance, plotValidationCurve
from load_data import PenDigitsDataset, SpamBaseDataset
import matplotlib.pyplot as plt


def experiment(pipe, dataset):
    num_neighbors = range(1, 50)
    train_sizes = np.linspace(0.1, 1, 40, endpoint=True)
    best_estimator = pipe
    alphas = [10 ** -x for x in np.arange(0, 5, 0.25)]

    makeAndPlotLearningCurve(best_estimator, "decisionTree", dataset.xtrain, dataset.ytrain,
                             train_sizes, "accuracy", "Accuracy", "ANN", dataset.name, "DefaultBase")
    makeAndPlotLearningCurve(best_estimator, "decisionTree", dataset.xtrain, dataset.ytrain,
                             train_sizes, "balanced_accuracy", "Accuracy", "ANN", dataset.name, "BaseBalanced")
    makeAndPlotLearningCurve(best_estimator, "decisionTree", dataset.xtrain, dataset.ytrain,
                             train_sizes, "f1_micro", "F1 Score", "ANN", dataset.name, "BaseF1")
    print(alphas)

    plotValidationCurve(best_estimator, "KNN", dataset.xtrain, dataset.ytrain,
                        "ann__alpha", alphas, "accuracy", None, "Accuracy", "ANN", dataset.name, "alpha")
    plotValidationCurve(best_estimator, "KNN", dataset.xtrain, dataset.ytrain, "ann__alpha",
                        alphas, "f1_micro", None, "F1 Score", "ANN", dataset.name, "alphaF1")
    pipe.fit(dataset.xtrain, dataset.ytrain)

    plotConfusionMatrix(best_estimator, dataset.xtest,
                        dataset.ytest, dataset.classes, "ANN", dataset.name)

    num_hiddens = [(100,), (100, 100), (100, 100, 100), (100, 100, 100, 100)]
    size_hidden = [(16,), (32,), (64,), (128,)]
    plt.clf()
    for layer, hidden in enumerate(num_hiddens):
        params = {'ann__hidden_layer_sizes': hidden}
        pipe.set_params(**params)
        pipe.fit(dataset.xtrain, dataset.ytrain)
        plt.plot(pipe["ann"].validation_scores_, label=str(layer+1))
    plt.title("Performance for varying Number of Hidden Layers")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'ANN/{dataset.name}/numLayers.png')

    plt.clf()
    for layer, hidden in enumerate(num_hiddens):
        params = {'ann__hidden_layer_sizes': hidden}
        pipe.set_params(**params)
        pipe.fit(dataset.xtrain, dataset.ytrain)
        plt.plot(pipe["ann"].loss_curve_, label=str(layer))
    plt.title("Loss curve for varying Number of Hidden Layers")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(f'ANN/{dataset.name}/numLayersLoss.png')

    plt.clf()
    for layer, hidden in enumerate(size_hidden):
        params = {'ann__hidden_layer_sizes': hidden}
        pipe.set_params(**params)
        pipe.fit(dataset.xtrain, dataset.ytrain)
        plt.plot(pipe["ann"].validation_scores_, "o-", label=str(hidden[0]))
    plt.title("Performance for varying size of hidden layer")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'ANN/{dataset.name}/sizeLayers.png')

    plt.clf()
    for layer, hidden in enumerate(num_hiddens):
        params = {'ann__hidden_layer_sizes': hidden}
        pipe.set_params(**params)
        pipe.fit(dataset.xtrain, dataset.ytrain)
        plt.plot(pipe["ann"].loss_curve_, "o-", label=str(hidden[0]))
    plt.title("Performance for varying size of hidden layer")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(f'ANN/{dataset.name}/sizeLayersLoss.png')
    #plotValidationCurve(best_estimator, "KNN", dataset.xtrain, dataset.ytrain, "ann__hidden_layer_sizes", hiddens , "accuracy", None, "Accuracy", "ANN", dataset.name, "numLayers")
    #plotValidationCurve(best_estimator, "KNN", dataset.xtrain, dataset.ytrain, "ann__hidden_layer_sizes", size_hidden, "accuracy", None, "Accuracy", "ANN", dataset.name, "sizeLayers")


def plotAnnCurves(mlp, X_train, Y_train,):
    epochs = 5000
    for epoch in range(1, epochs):
        mlp.fit(X_train, Y_train)
        Y_pred = mlp.predict(X_train)
        curr_train_score = mean_squared_error(
            Y_train, Y_pred)  # training performances
        Y_pred = mlp.predict(X_valid)
        curr_valid_score = mean_squared_error(
            Y_valid, Y_pred)  # validation performances
        training_mse.append(curr_train_score)  # list of training perf to plot
        validation_mse.append(curr_valid_score)  # list of valid perf to plot
    plt.plot(training_mse, label="train")
    plt.plot(validation_mse, label="validation")
    plt.legend()


def main():

    Pendataset = PenDigitsDataset()

    ann = neural_network.MLPClassifier(
        max_iter=5000, early_stopping=True, random_state=42,)
    dataset = SpamBaseDataset()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("ann", ann)])
    #neighborsComplexity(pipe, dataset)
    #neighborsComplexity(pipe, Pendataset)
    #distanceMetric(pipe, Pendataset)
    #distanceMetric(pipe, dataset)
    d = dataset.xtrain.shape[1]
    hiddens = [(h,) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]
    print(hiddens)

    experiment(pipe, dataset)
    experiment(pipe, Pendataset)


if __name__ == "__main__":
    main()
