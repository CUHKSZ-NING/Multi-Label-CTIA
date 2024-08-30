# Multi-Label CTIA

Code and dataset for manuscript `Linked Meta Classifiers for Multi-Label Disease Detection Under Class Imbalance`.

* Required Python 3 packages:
    1. `numpy==1.21.5`
    2. `sklearn` (https://github.com/scikit-learn/scikit-learn)
    3. `scipy==1.7.3`
    4. `iterative-stratification` (https://github.com/trent-b/iterative-stratification)

* Optional Python 3 packages: 
    1. `imblearn` (https://github.com/scikit-learn-contrib/imbalanced-learn)

* MetaLink is compatible with most sklearn APIs but is not strictly tested.

* Import: `from MetaLink import MetaLink`

* Train: `fit(X, y)`, with target $\textbf{y}_i \in (0, 1)^l$ as the labels. 

* Predict: `predict(X)` (hard prediction), `predict_proba(X)` (probalistic prediction).

* Parameters: 
    1. `base_estimators`: a list of multiple (at least one) classifier objects with `predict_proba()` function
    2. `n_loops`: an integer representing the number of iterations
    4. `meta_estimator`: classifier object with `predict_proba()` function, "meta classifier for MetaLink"
