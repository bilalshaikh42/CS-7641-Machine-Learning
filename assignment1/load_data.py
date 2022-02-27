import pandas as pd
import numpy as np
from sklearn import model_selection


class PenDigitsDataset(object):
    def __init__(self,
                 path="../data/pen_digits.csv"):
        self.name = "PenDigits"
        self.path = path
        self.data = pd.read_csv(self.path, header=None)
        print(self.data.head())
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.feats = np.array(self.data.iloc[:, 0:-1])
        self.labels = np.array(self.data.iloc[:, -1])
        print(self.feats.shape)
        print(self.labels.shape)
        self.xtrain, self.xtest, self.ytrain, self.ytest = model_selection.train_test_split(
            self.feats, self.labels, shuffle=True, test_size=0.2, random_state=42, stratify=self.labels)
        self.scorer = "accuracy"
        self.score_label = "Accuracy"
        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class SpamBaseDataset(object):
    def __init__(self,
                 path="../data/spambase.csv"):
        self.name = "SpamBase"
        self.path = path
        self.data = pd.read_csv(self.path, header=None)
        print(self.data.head())
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.feats = np.array(self.data.iloc[:, 0:-1])
        self.labels = np.array(self.data.iloc[:, -1])
        print(self.feats.shape)
        print(self.labels.shape)
        self.xtrain, self.xtest, self.ytrain, self.ytest = model_selection.train_test_split(
            self.feats, self.labels, shuffle=True, test_size=0.2, random_state=42, stratify=self.labels)
        self.scorer = "f1_micro"
        self.score_label = "F1 score"
        self.classes = ["ham", "spam"]


class DefaultDataset(object):
    def __init__(self) -> None:
        self.name = "CreditDefault"
        self.path = "../data/default.xls"
        self.data = pd.read_excel(self.path)
        self.feats = np.array(self.data.iloc[:, 0:-1])
        self.labels = np.array(self.data.iloc[:, -1])
        self.xtrain, self.xtest, self.ytrain, self.ytest = model_selection.train_test_split(
            self.feats, self.labels, shuffle=True, test_size=0.2, random_state=42, stratify=self.labels)
        self.scorer = "balanced_accuracy"
        self.score_label = "Accuracy"


if(__name__ == "__main__"):
    dataset = PenDigitsDataset()
    print(dataset.xtrain.shape)
    print(dataset.ytrain.shape)
