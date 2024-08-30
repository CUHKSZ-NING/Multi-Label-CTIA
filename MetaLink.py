import numpy as np
from sklearn.metrics import average_precision_score
from imblearn.metrics import geometric_mean_score
from copy import deepcopy
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix as MLCM
from imblearn.ensemble import BalancedBaggingClassifier as Bagging, EasyEnsembleClassifier as Easy


class MetaLink(object):
    def __init__(self, base_estimators=None, meta_estimator=None, n_loops=10, weight_decay=0.9):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.n_loops = n_loops
        self.weight_decay = weight_decay
        self.base_classifiers = {}
        self.meta_classifiers = {}
        self.n_labels = -1
        self.n_estimators = len(self.base_estimators)

    def fit(self, X, y):
        self.n_labels = len(y[0])
        for i in range(0, self.n_loops):
            self.meta_classifiers[i] = {}
            for j in range(0, self.n_labels):
                self.meta_classifiers[i][j] = deepcopy(self.meta_estimator)
        n_samples = len(X)
        predictions = {}
        meta_features = None
        n_folds = 5

        for i in range(0, self.n_loops + 1):
            print(i)
            predictions[i] = {}
            self.base_classifiers[i] = {}
            for j in range(0, self.n_labels):
                predictions[i][j] = {}
                self.base_classifiers[i][j] = {}
                for k in range(0, self.n_estimators):
                    predictions[i][j][k] = np.zeros(n_samples)
                    self.base_classifiers[i][j][k] = []

            if i > 0:
                stat = {}
                for j in range(0, self.n_labels):
                    stat[j] = {}
                    for k in range(0, self.n_estimators):
                        stat[j][k] = None
                        for m in range(0, i):
                            if stat[j][k] is None:
                                stat[j][k] = np.copy([predictions[m][j][k]])
                            else:
                                stat[j][k] = np.concatenate((stat[j][k], [predictions[m][j][k]]))

                meta_features = None

                for k in range(0, self.n_estimators):
                    for j in range(0, self.n_labels):
                        if meta_features is None:
                            meta_features = deepcopy([np.mean(stat[j][k], axis=0)])
                        else:
                            meta_features = np.concatenate((meta_features, [np.mean(stat[j][k], axis=0)]))

                    meta_features = np.concatenate((meta_features,
                                                    [np.sum(meta_features[-1 * self.n_labels:], axis=0)]))
                    uncertainty = np.zeros(n_samples)
                    for p in range(0, n_samples):
                        for q in range(0, self.n_labels):
                            value = meta_features[-2 - q][p]
                            uncertainty[p] += np.min([1 - value, value])
                    meta_features = np.concatenate((meta_features, [uncertainty]))

                    for j in range(0, self.n_labels):
                        meta_features = np.concatenate((meta_features, [np.std(stat[j][k], axis=0)]))
                meta_features = meta_features.T

                for j in range(0, self.n_labels):
                    # self.meta_classifier[j].fit(np.concatenate(X, meta_features), y[..., j])
                    self.meta_classifiers[i - 1][j].fit(meta_features, y[..., j])

                if i == self.n_loops:
                    break

            skf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True)

            for index_train, index_test in skf.split(X, y):
                X_train = np.copy(X[index_train])
                y_train = np.copy(y[index_train])
                X_val = np.copy(X[index_test])
                y_val = np.copy(y[index_test])

                for j in range(0, self.n_labels):
                    for k in range(0, self.n_estimators):
                        model = deepcopy(self.base_estimators[k])

                        if i == 0:
                            model.fit(X_train, y_train[..., j])
                            model.predict_proba(X_val)
                            predictions[i][j][k][index_test] = model.predict_proba(X_val)[..., 1]

                        else:
                            X_train_new = np.concatenate((X_train, meta_features[index_train]), axis=1)
                            model.fit(X_train_new, y_train[..., j])
                            X_val_new = np.concatenate((X_val, meta_features[index_test]), axis=1)
                            predictions[i][j][k][index_test] = model.predict_proba(X_val_new)[..., 1]

                        self.base_classifiers[i][j][k].append(deepcopy(model))

    def predict(self, X):
        for i in range(0, self.n_loops):
            pass
        pass

    def predict_proba(self, X):
        y_pred_proba = np.zeros((len(X), self.n_labels))

        predictions = {}
        for i in range(0, self.n_loops + 1):
            predictions[i] = {}
            for j in range(0, self.n_labels):
                predictions[i][j] = {}

        for i in range(0, self.n_loops + 1):
            meta_features = None

            if i > 0:
                stat = {}
                for j in range(0, self.n_labels):
                    stat[j] = {}
                    for k in range(0, self.n_estimators):
                        stat[j][k] = None
                        for m in range(0, i):
                            if stat[j][k] is None:
                                stat[j][k] = np.copy([predictions[m][j][k]])
                            else:
                                stat[j][k] = np.concatenate((stat[j][k], [predictions[m][j][k]]))

                meta_features = None

                for k in range(0, self.n_estimators):
                    for j in range(0, self.n_labels):
                        if meta_features is None:
                            meta_features = deepcopy([np.mean(stat[j][k], axis=0)])
                        else:
                            meta_features = np.concatenate((meta_features, [np.mean(stat[j][k], axis=0)]))

                    meta_features = np.concatenate((meta_features,
                                                    [np.sum(meta_features[-1 * self.n_labels:], axis=0)]))
                    uncertainty = np.zeros(len(X))
                    for p in range(0, len(X)):
                        for q in range(0, self.n_labels):
                            value = meta_features[-2 - q][p]
                            uncertainty[p] += np.min([1 - value, value])
                    meta_features = np.concatenate((meta_features, [uncertainty]))

                    for j in range(0, self.n_labels):
                        meta_features = np.concatenate((meta_features, [np.std(stat[j][k], axis=0)]))
                meta_features = meta_features.T

                for j in range(0, self.n_labels):
                    y_pred_proba[..., j] += (
                            self.meta_classifiers[i - 1][j].predict_proba(meta_features)[..., 1] / self.n_loops)

                if i == self.n_loops:
                    return y_pred_proba

            for j in range(0, self.n_labels):
                for k in range(0, self.n_estimators):
                    prediction = None
                    n_folds = len(self.base_classifiers[i][j][k])
                    for p in range(0, n_folds):
                        if i == 0:
                            if prediction is None:
                                prediction = deepcopy(
                                    self.base_classifiers[i][j][k][p].predict_proba(X)[..., 1] / n_folds)
                            else:
                                prediction += deepcopy(
                                    self.base_classifiers[i][j][k][p].predict_proba(X)[..., 1] / n_folds)
                        else:
                            if prediction is None:
                                prediction = deepcopy(self.base_classifiers[i][j][k][p].predict_proba(
                                    np.concatenate((X, meta_features), axis=1))[..., 1] / n_folds)
                            else:
                                prediction += deepcopy(self.base_classifiers[i][j][k][p].predict_proba(
                                    np.concatenate((X, meta_features), axis=1))[..., 1] / n_folds)

                    predictions[i][j][k] = deepcopy(prediction)



