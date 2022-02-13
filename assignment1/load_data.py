import pandas as pd
import numpy as np
from sklearn import model_selection


class PenDigitsDataset(object):
    def __init__(self,
                 path="../data/pen_digits.csv"):

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
            self.feats, self.labels, shuffle=True, test_size=0.3, random_state=42, stratify=self.labels)

class SpamBaseDataset(object):
    def __init__(self,
                 path="../data/spambase.data"):

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
            self.feats, self.labels, shuffle=True, test_size=0.3, random_state=42, stratify=self.labels)


if(__name__ == "__main__"):
    dataset = PenDigitsDataset()
    print(dataset.xtrain.shape)
    print(dataset.ytrain.shape)
