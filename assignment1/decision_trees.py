
import warnings
import sklearn
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from plots import  plotConfusionMatrix, plotPerformance, makeAndPlotLearningCurve, plotValidationCurve
from load_data import DefaultDataset, PenDigitsDataset, SpamBaseDataset
from sklearn import  tree
from sklearn.model_selection import GridSearchCV as grid_search
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from multiprocessing import Pool


class DecisionTree(object):
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,

                 ) -> None:

        self.dt = tree.DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            
            class_weight=class_weight
        )


def complexity(pipe, dataset):
    max_depth = range(1, 50)
    score_train = []
    score_test = []
    prec_train = []
    prec_test = []

    for depth in max_depth:
        pipe.set_params(dt__max_depth=depth)
        print(dataset.xtrain.shape)
        print(dataset.ytrain.shape)
        pipe.fit(dataset.xtrain, dataset.ytrain)
        y_train_pred = pipe.predict(dataset.xtrain)
        y_test_pred = pipe.predict(dataset.xtest)
        print(dataset.ytrain)
        print(y_test_pred)
        print(y_train_pred)

        score_train.append(accuracy_score(dataset.ytrain, y_train_pred))
        score_test.append(accuracy_score(dataset.ytest, y_test_pred))
        prec_train.append(precision_score(
            dataset.ytrain, y_train_pred, average='weighted'))
        prec_test.append(precision_score(
            dataset.ytest, y_test_pred, average='weighted'))

        print(score_train)
        print(score_test)
    plotPerformance(max_depth, score_test, score_train, "Max Depth", "Accuracy",
                    "decisionTree", dataset.name, f'maxDepth', "Model Performance vs Complexity (depth)")


def leavesComplexity(pipe, dataset):

    with Pool() as p:
        depths = [10, 20, 30, 40]
        p.starmap(leavesDepthComplexity, [
                  (pipe, dataset, depth) for depth in depths])


def leavesDepthComplexity(pipe, dataset, depth):
    max_leaves = range(10, 500, 5)

    score_train = []
    score_test = []
    prec_train = []
    prec_test = []
    for leaves in max_leaves:
        print("depth: ", depth, "leaves: ", leaves)
        pipe.set_params(dt__max_depth=depth, dt__max_leaf_nodes=leaves)

        pipe.fit(dataset.xtrain, dataset.ytrain)
        y_train_pred = pipe.predict(dataset.xtrain)
        y_test_pred = pipe.predict(dataset.xtest)

        score_train.append(accuracy_score(dataset.ytrain, y_train_pred))
        score_test.append(accuracy_score(dataset.ytest, y_test_pred))
        prec_train.append(precision_score(
            dataset.ytrain, y_train_pred, average='weighted'))
        prec_test.append(precision_score(
            dataset.ytest, y_test_pred, average='weighted'))

    plotPerformance(max_leaves, score_test, score_train, "Max Leaf Nodes", "Accuracy", "decisionTree",
                    dataset.name, f'Depth-{depth}-Leaves-{max_leaves[-1]}', "Model Performance vs Complexity (leaf count)")

    params = {
        'dt__max_depth': depth,
        'dt__max_leaf_nodes': max_leaves[np.argmax(score_test)],
    }
    learningCurve(pipe, dataset, params, name="{d}depthleaves".format(d=depth))


def criteria(pipe, dataset):
    score_train = []
    score_test = []
    prec_train = []
    prec_test = []
    criterions = ["gini", "entropy"]
    for criteria in criterions:
        pipe.set_params(dt__criterion=criteria)

        pipe.fit(dataset.xtrain, dataset.ytrain)
        y_train_pred = pipe.predict(dataset.xtrain)
        y_test_pred = pipe.predict(dataset.xtest)
        print(dataset.ytrain)
        print(y_test_pred)
        print(y_train_pred)

        score_train.append(accuracy_score(dataset.ytrain, y_train_pred))
        score_test.append(accuracy_score(dataset.ytest, y_test_pred))
        prec_train.append(precision_score(
            dataset.ytrain, y_train_pred, average='weighted'))
        prec_test.append(precision_score(
            dataset.ytest, y_test_pred, average='weighted'))

        print(score_train)
        print(score_test)

    plt.clf()
    fig, ax = plt.subplots()

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Training", "Test"])
    width = 0.35
    ax.bar([0-width/2, 1-width/2], [score_train[0],
           score_test[0]], label="Gini", width=width)
    ax.bar([0+width/2, 1+width/2], [score_train[1],
           score_test[1]], label="Entropy", width=width)
    plt.legend(loc='best')
    plt.title(f'Model Performance for different splitting criteria ({dataset.name})')
    plt.xlabel("Split Criteria")
    plt.ylabel("Accuracy")
    fig.savefig(f'decisionTree/{dataset.name}criteria.png')


def scorerFunc(y_true, y_pred):
    weights = compute_sample_weight('balanced', y_true)
    return accuracy_score(y_true, y_pred, sample_weight=weights)

# from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

def exploreTree(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    # count number of trues in is_leaves
    numLeaves = np.sum(is_leaves)
    print("Number of leaves: %d" % numLeaves)

    print(
        "The binary tree structure has {n} nodes and has {l} leaves.".format(n=n_nodes, l=numLeaves))


def learningCurve(pipe, dataset, params=None, name="Base"):
    train_sizes = np.linspace(0.1, 1, 40, endpoint=True)
    makeAndPlotLearningCurve(pipe, "decisionTree",dataset.xtrain,dataset.ytrain,train_sizes, "accuracy", "Accuracy", "Decision Tree", dataset.name, "DefaultBase")
    max_depths = np.arange(1, 51, 1)
    max_leaf_nodes = np.arange(2, 10, 1)
    if params is None:
        params = {'dt__criterion': ['gini', 'entropy'], 'dt__max_depth': max_depths,
                  'dt__class_weight': ['balanced', None]}  # 'dt__max_leaf_nodes': max_leaf_nodes}

        scorer = make_scorer(scorerFunc)

        grid_search = ms.GridSearchCV(
            pipe, n_jobs=8, param_grid=params, refit=True, verbose=10, cv=5, scoring=dataset.scorer)
        
        grid_search.fit(dataset.xtrain, dataset.ytrain,)
        exploreTree(grid_search.best_estimator_.steps[1][1])

        best_estimator = grid_search.best_estimator_
        best_estimator.fit(
            dataset.xtrain, dataset.ytrain)
    else:
        best_estimator = pipe
        best_estimator.set_params(**params)
        best_estimator.fit(dataset.xtrain, dataset.ytrain)

  
    best_params= best_estimator.get_params()
    #makeAndPlotLearningCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain,train_sizes, "accuracy", "Accuracy", "Decision Tree", dataset.name, "Base")
    #makeAndPlotLearningCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain,train_sizes, "balanced_accuracy", "Accuracy", "Decision Tree", dataset.name, "BaseBalanced")
    #makeAndPlotLearningCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain,train_sizes, "f1_micro", "F1 Score", "Decision Tree", dataset.name, "BaseF1")
    #plotValidationCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain, "dt__max_depth", np.arange(1, 51, 1), "accuracy",None, "Accuracy", "Decision Tree", dataset.name, "MaxDepth")
    #plotValidationCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain, "dt__max_depth", np.arange(1, 51, 1), "balanced_accuracy",None, "Accuracy", "Decision Tree", dataset.name, "MaxDepthBalanced")
    #plotValidationCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain, "dt__max_depth", np.arange(1, 51, 1), "f1_weighted",None, "F1 Score", "Decision Tree", dataset.name, "MaxDepthF1")
    #plotValidationCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain, "dt__max_leaf_nodes", np.arange(2, 500, 1), "accuracy",None, "Accuracy", "Decision Tree", dataset.name, "MaxLeaves")
    #plotValidationCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain, "dt__min_samples_leaf", np.arange(1, 50, 1), "accuracy",None, "Accuracy", "Decision Tree", dataset.name, "MinSamplesLeaf")
    #plotValidationCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain, "dt__criterion", ['gini', 'entropy'], "accuracy",None, "Accuracy", "Decision Tree", dataset.name, "Criterion")
    path = best_estimator["dt"].cost_complexity_pruning_path(dataset.xtrain, dataset.ytrain)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    best_estimator.set_params()
    #plotValidationCurve(best_estimator, "decisionTree",dataset.xtrain,dataset.ytrain, "dt__ccp_alpha", ccp_alphas, "accuracy",None, "Accuracy", "Decision Tree", dataset.name, "CCPAlpha")
    #print("Best Params")
    #print(best_params)
    path = "{}/{}/params.txt".format("Decision Tree",dataset.name,)

    #with open(path, "w") as f:
    #    f.write(str(best_params))
    plotConfusionMatrix(best_estimator, dataset.xtest, dataset.ytest, dataset.classes, "Decision Trees", dataset.name)

def main():
    warnings.simplefilter("ignore", sklearn.exceptions.DataConversionWarning)
    
    Pendataset = PenDigitsDataset()
    dataset = SpamBaseDataset()
    dt = DecisionTree()
    default = DefaultDataset()
    pipe = Pipeline([("scaler", StandardScaler()),
                    ('dt', dt.dt)])

    #complexity(pipe, dataset)
    #criteria(pipe, dataset)
   
    
    #learningCurve(pipe, default)
    learningCurve(pipe, dataset)
    #leavesComplexity(pipe, dataset)

    #complexity(pipe, Pendataset)
    #criteria(pipe, Pendataset)
    learningCurve(pipe, Pendataset)
    #leavesComplexity(pipe, Pendataset)


if __name__ == "__main__":
    main()
