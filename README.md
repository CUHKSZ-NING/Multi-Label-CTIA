# MEFAClassifier

Code for manuscript `Towards Multi-Label Disease Detection Using Imbalanced Tongue Data`.

* Required Python 3 packages:
    1. numpy==1.21.5
    2. scipy==1.7.3

* Optional Python 3 packages: 
    1. `sklearn` (https://github.com/scikit-learn/scikit-learn)
    2. `imblearn` (https://github.com/scikit-learn-contrib/imbalanced-learn)

* MEFA is compatible with most sklearn APIs but is not strictly tested.

* Import: `from MEFAClassifier import MEFAClassifier`

* Train: `fit(X, y)`, with target $y_i \in \{0, 1\}$ as the labels (Note that $y = 0$ and $y = 1$ denote majoirty and minority class labels, respectively). 

* Predict: `predict(X)` (hard prediction), `predict_proba(X)` (probalistic prediction).

* Parameters: 
    1. `base_estimators`: classifier object with `predict_proba()` function, "candidate classifier $f(\cdot)$ for SPISEClassifier"
    2. `n_estimators`: int, `default=10`, "the number of training rounds $\alpha$ for SPISEClassifier"
    3. `n_subsets`: int, `default=3`, "the number of subsets $q$ selected in each round"
    4. `n_entries`: int, `default=3`, "the number of non-zeros entries in each colomn of the projection matrix $\textbf{M}$"
