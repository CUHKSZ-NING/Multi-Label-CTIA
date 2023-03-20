# MEFAClassifier

Code for manuscript `Towards Multi-Label Disease Detection Using Imbalanced Tongue Data`.

* Required Python 3 packages:
    1. `numpy==1.21.5`
    2. `sklearn` (https://github.com/scikit-learn/scikit-learn)
    3. `scipy==1.7.3`
    4. `iterative-stratification` (https://github.com/trent-b/iterative-stratification)

* Optional Python 3 packages: 
    1. `imblearn` (https://github.com/scikit-learn-contrib/imbalanced-learn)

* MEFA is compatible with most sklearn APIs but is not strictly tested.

* Import: `from MEFAClassifier import MEFAClassifier`

* Train: `fit(X, y)`, with target $\textbf{y}_i \in \{0, 1\}^l$ as the labels. 

* Predict: `predict(X)` (hard prediction), `predict_proba(X)` (probalistic prediction).

* Parameters: 
    1. `base_estimator`: classifier object with `predict_proba()` function, "candidate classifier $f^{(1)}(\cdot)$ for MEFAClassifier"
    2. `auxiliary_estimator`: classifier object with `predict_proba()` function, "candidate classifier $f^{(2)}(\cdot)$ for MEFAClassifier"
    3. `regressor`: regressor object with `predict()` function, "regressor $f^{r}(\cdot)$ for MEFAClassifier"
    4. `meta_estimator`: classifier object with `predict_proba()` function, "meta classifier for MEFAClassifier"
