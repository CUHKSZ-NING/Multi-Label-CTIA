import numpy as np
from sklearn.metrics import average_precision_score
from imblearn.metrics import geometric_mean_score
from copy import deepcopy
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix as MLCM
from imblearn.ensemble import BalancedBaggingClassifier as Bagging, EasyEnsembleClassifier as Easy


class MetaFeatures(object):
    def __init__(self, base_estimator, auxiliary_estimator=None, meta_estimator=None, regressor=None,
                 health_assessment=True, augment=True):
        self.base_estimator = base_estimator
        self.auxiliary_estimator = auxiliary_estimator
        self.health_assessment = health_assessment
        if meta_estimator is None:
            self.meta_estimator = NB(priors=[0.5, 0.5])
        else:
            self.meta_estimator = deepcopy(meta_estimator)
        self.n_labels = 0
        self.base_models = {}
        self.auxiliary_models = {}
        self.meta_models = {}
        self.regressor = regressor
        self.regression_models = {}
        self.health_models = {}
        self.augment = augment
        # self.m_trees = m_trees
    
    def fit(self, X, y):
        self.n_labels = len(y[0, ...])
        n_folds = 5
        meta_data = {'data': None, 'target': None}
        
        for i in range(0, n_folds):
            self.base_models[i] = {}
            self.auxiliary_models[i] = {}
            self.health_models = {}
        fold_counter = 0
        
        skf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True)
        
        for index_train, index_test in skf.split(X, y):
            X_train = np.copy(X[index_train])
            y_train = np.copy(y[index_train])
            X_val = np.copy(X[index_test])
            y_val = np.copy(y[index_test])
            
            if meta_data['target'] is None:
                meta_data['target'] = deepcopy(y_val)
            else:
                meta_data['target'] = np.concatenate((meta_data['target'], y_val))
            
            meta_prediction = None  # prediction meta features
            
            for i in range(0, self.n_labels):
                model = deepcopy(self.base_estimator)
                model.fit(X_train, y_train[..., i])
                self.base_models[fold_counter][i] = deepcopy(model)
                label_pred_proba = [model.predict_proba(X_val)[..., 1]]
                
                if self.auxiliary_estimator is not None:
                    model = deepcopy(self.auxiliary_estimator)
                    model.fit(X_train, y_train[..., i])
                    self.auxiliary_models[fold_counter][i] = deepcopy(model)
                    label_pred_proba = np.concatenate((label_pred_proba, [model.predict_proba(X_val)[..., 1]]))
                
                if meta_prediction is None:
                    meta_prediction = deepcopy(label_pred_proba)
                else:
                    meta_prediction = np.concatenate((meta_prediction, label_pred_proba))
            
            if self.health_assessment:
                model = deepcopy(self.base_estimator)
                model.fit(X_train, np.max(y_train, axis=1))
                self.health_models[fold_counter] = {}
                self.health_models[fold_counter]['main'] = deepcopy(model)
                label_pred_proba = [model.predict_proba(X_val)[..., 1]]
                meta_prediction = np.concatenate((meta_prediction, label_pred_proba))
                if self.auxiliary_estimator is not None:
                    model = deepcopy(self.auxiliary_estimator)
                    model.fit(X_train, np.max(y_train, axis=1))
                    self.health_models[fold_counter]['auxiliary'] = deepcopy(model)
                    label_pred_proba = [model.predict_proba(X_val)[..., 1]]
                    meta_prediction = np.concatenate((meta_prediction, label_pred_proba))
            
            if self.regressor is not None:
                model = deepcopy(self.regressor)
                model.fit(X_train, np.sum(y_train, axis=1))
                self.regression_models[fold_counter] = deepcopy(model)
                value_pred_proba = model.predict(X_val)
                meta_prediction = np.concatenate((meta_prediction, [value_pred_proba]))
            
            fold_counter += 1
            
            meta_prediction = meta_prediction.T
            if meta_data['data'] is None:
                meta_data['data'] = deepcopy(meta_prediction)
            else:
                meta_data['data'] = np.concatenate((meta_data['data'], meta_prediction))
            
        for i in range(0, self.n_labels):
            model = deepcopy(self.meta_estimator)
            # model.n_estimators = self.m_trees
            model.fit(meta_data['data'], meta_data['target'][..., i])
            self.meta_models[i] = deepcopy(model)
        
        # y_pred_proba = self.predict_proba(X)
        # print('%.3f <---- AUC' % roc_auc_score(y, y_pred_proba, average='macro'))
        # y_pred = self.predict(X)
        # print('%.3f <---- MaF1' % MacroF1(y, y_pred))
        #
        # print('')
    
    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        y_pred = np.zeros((len(X), self.n_labels), dtype=int)
        for i in range(0, self.n_labels):
            y_pred[..., i][np.where(y_pred_proba[..., i] >= 0.5)] = 1
        
        return y_pred
    
    def predict_proba(self, X):
        X_meta = None
        fold_counter = 0
        meta_prediction = None  # prediction meta features
        y_pred_proba = np.zeros((len(X), self.n_labels))
        
        for key in self.base_models:
            fold_counter += 1
            for i in range(0, self.n_labels):
                label_pred_proba = [self.base_models[key][i].predict_proba(X)[..., 1]]
                if self.auxiliary_estimator is not None:
                    label_pred_proba = np.concatenate(
                        (label_pred_proba, [self.auxiliary_models[key][i].predict_proba(X)[..., 1]]))
                
                if meta_prediction is None:
                    meta_prediction = deepcopy(label_pred_proba)
                else:
                    meta_prediction = np.concatenate((meta_prediction, label_pred_proba))
            
            if self.health_assessment:
                label_pred_proba = [self.health_models[key]['main'].predict_proba(X)[..., 1]]
                meta_prediction = np.concatenate((meta_prediction, label_pred_proba))
                if self.auxiliary_estimator is not None:
                    label_pred_proba = [self.health_models[key]['auxiliary'].predict_proba(X)[..., 1]]
                    meta_prediction = np.concatenate((meta_prediction, label_pred_proba))
            
            if self.regressor is not None:
                value_pred_proba = self.regression_models[key].predict(X)
                meta_prediction = np.concatenate((meta_prediction, [value_pred_proba]))

            meta_prediction = meta_prediction.T
            if X_meta is None:
                X_meta = deepcopy(meta_prediction)
            else:
                X_meta += meta_prediction
            meta_prediction = None
        
        X_meta /= fold_counter
        
        for i in range(0, self.n_labels):
            y_pred_proba[..., i] = self.meta_models[i].predict_proba(X_meta)[..., 1]
        
        return y_pred_proba


def MacroF1(y_true, y_pred):
    matrix = MLCM(y_true, y_pred)
    matrix = matrix.astype(float)
    n_labels = len(matrix)
    MaF1 = 0
    for c in range(0, n_labels):
        matrix[c][0, ...] /= sum(matrix[c][0, ...])
        matrix[c][1, ...] /= sum(matrix[c][1, ...])
        recall = matrix[c][1, 1]
        precision = matrix[c][1, 1] / sum(matrix[c][..., 1])
        if precision != precision:
            precision = 0
        F1 = 2 * recall * precision / (recall + precision)
        if F1 != F1:
            F1 = 0
        MaF1 += F1 / n_labels
        print('%.3f ' % F1, end='')
    print('')
    
    return MaF1
