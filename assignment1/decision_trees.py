
import warnings
import sklearn
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from load_data import PenDigitsDataset, SpamBaseDataset
from sklearn import grid_search, tree
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
            min_impurity_split=min_impurity_split,
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

    plt.plot(max_depth, score_test, 'o-',  label='Test Accuracy')
    plt.plot(max_depth, score_train, 'o-',  label='Training Accuracy')

    plt.ylabel('Model Performance')
    plt.xlabel('Max Tree Depth')

    plt.title("Performance of Decision Tree based on complexity")
    plt.legend(loc='best')
    plt.tight_layout()
    print(score_test)
    print(score_train)
    plt.savefig("decisionTree/depthCurve.png")



def leavesComplexity(pipe, dataset):

    with Pool() as p:
        depths=[10,20,30,40]
        p.starmap(leavesDepthComplexity, [(pipe, dataset, depth) for depth in depths])


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
    plt.clf()
    plt.plot(max_leaves, score_test, 'o-',  label='Test Accuracy')
    plt.plot(max_leaves, score_train, 'o-',  label='Training Accuracy')

    plt.ylabel('Model Performance')
    plt.xlabel('Max Leaves Count')

    plt.title("Performance of Decision Tree based on complexity")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("decisionTree/{d}depthleavesCurve.png".format(d=depth))
    params= {
        'dt__max_depth': depth,
        'dt__max_leaf_nodes': max_leaves[np.argmax(score_test)],
    }
    learningCurve(pipe, dataset, params, name="{d}depthleaves".format(d=depth))

def criteria(pipe, dataset):
    score_train = []
    score_test = []
    prec_train = []
    prec_test = []

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
    fig.savefig("decisionTree/criteria.png")


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


def learningCurve(pipe, dataset, params=None, name=None):

    max_depths = np.arange(1, 51, 1)
    max_leaf_nodes = np.arange(2, 10, 1)
    if params is None:
        params = {'dt__criterion': ['gini', 'entropy'], 'dt__max_depth': max_depths,
                'dt__class_weight': ['balanced', None]}  # 'dt__max_leaf_nodes': max_leaf_nodes}

        scorer = make_scorer(scorerFunc)

        grid_search = ms.GridSearchCV(
            pipe, n_jobs=8, param_grid=params, refit=True, verbose=10, cv=5, scoring=scorer,
        )
        grid_search.fit(dataset.xtrain, dataset.ytrain,)
        exploreTree(grid_search.best_estimator_.steps[1][1])

        best_estimator = grid_search.best_estimator_
        best_estimator.set_params(dt__max_depth=10)
        best_estimator.fit(
            dataset.xtrain, dataset.ytrain)
    else:
        best_estimator = pipe
        best_estimator.set_params(**params)
        best_estimator.fit(dataset.xtrain, dataset.ytrain)

    train_sizes = np.append(np.linspace(0.05, 0.1, 20, endpoint=False),
                            np.linspace(0.1, 1, 20, endpoint=True))
    train_sizes = np.linspace(0.1, 1, 40, endpoint=True)
    train_sizes, train_scores, test_scores = ms.learning_curve(
        best_estimator,
        dataset.xtrain, dataset.ytrain, cv=5, n_jobs=8,
        verbose=10,

        train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    train_points = train_scores_mean
    test_points = test_scores_mean
    plt.clf()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2)

    plt.plot(train_sizes, train_points, 'o-', linewidth=1, markersize=4,
             label="Training score")
    plt.plot(train_sizes, test_points, 'o-', linewidth=1, markersize=4,
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig("decisionTree/{}learningCurve.png".format(name))


def main():
    warnings.simplefilter("ignore", sklearn.exceptions.DataConversionWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    # dataset = PenDigitsDataset()
    dataset = SpamBaseDataset()
    dt = DecisionTree()
    pipe = Pipeline([("scaler", StandardScaler()),
                    ('dt', dt.dt)])

    criterions = ["gini", "entropy"]
    # $complexity(pipe, dataset)
    #criteria(pipe, dataset)
    #paramters(pipe, dataset)
    leavesComplexity(pipe, dataset)


if __name__ == "__main__":
    main()
