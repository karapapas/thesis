import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class OptimizationMethods:

    # find optimal number of PCs
    @staticmethod
    def find_optimal_pcs_number(x, y):

        pca = PCA()
        # set the tolerance to a large value to make the example faster
        algorithm = GaussianNB()
        pipe = Pipeline(steps=[('pca', pca), ('algorithm', algorithm)])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        param_grid = {'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
        search = GridSearchCV(pipe, param_grid, n_jobs=-1)
        search.fit(x.values, y)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)

        # Plot the PCA spectrum
        pca.fit(x)

        fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(12, 6))

        # For each number of components, find the best classifier results
        results = pd.DataFrame(search.cv_results_)
        components_col = 'param_pca__n_components'
        best_clfs = results.groupby(components_col).apply(lambda g: g.nlargest(1, 'mean_test_score'))

        best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score', legend=False, ax=ax)

        ax.set_ylabel('Classification accuracy (val)')
        ax.set_xlabel('n_components')

        plt.xlim(-1, 15)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def hpo(x, y):

        # instantiate algorithms with default settings
        lr = LogisticRegression(random_state=7)
        dt = DecisionTreeClassifier(random_state=7)
        rf = RandomForestClassifier(random_state=7)
        sv = SVC(probability=True)
        gn = GaussianNB()
        mp = MLPClassifier(random_state=7, max_iter=200)
        kn = KNeighborsClassifier()

        model_params = {
            'Logistic Regression': (lr, {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                                         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}),
            'Decision Tree': (dt, {'criterion': ['gini', 'entropy'],
                                   'max_depth': np.arange(1, 50, 2)}),
            'Random Forest': (rf, {'n_estimators': np.arange(1, 15, 3),
                                   'criterion': ['gini', 'entropy'],
                                   'max_depth': np.arange(1, 50, 2)}),
            'Support Vector Classifier': (sv, {'C': np.arange(0, 5, 0.5),
                                               'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                               'degree': np.arange(1, 3, 1)}),
            'Gaussian Naive Bayes': (gn, {'var_smoothing': np.arange(0, 5, 0.1)}),
            'Multi-layer Perceptron': (mp, {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                            'solver': ['lbfgs', 'sgd', 'adam']}),
            'K Neighbors Classifier': (kn, {'n_neighbors': np.arange(1, 50, 1),
                                            'weights': ['uniform', 'distance'],
                                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']})
        }

        for (md, model) in model_params.items():
            search = GridSearchCV(model[0], model[1], n_jobs=-1)
            search.fit(x, y)
            print("Best CV Score for %s: %.2f Parameters: %s." % (md, search.best_score_, search.best_params_))
