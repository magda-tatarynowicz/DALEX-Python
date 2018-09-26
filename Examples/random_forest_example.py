# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def create_random_forest_model() : 
    X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                random_state=0)
    clf.fit(X, y)
    return(X, y, clf, ['col1','col2','col3','col4'])