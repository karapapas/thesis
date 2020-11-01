from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import time
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


class TrainingMethods:

    # takes a dataframe and a list of features to train on
    # returns a dictionary of trained models
    @staticmethod
    def train_models(x_train, y_train, x_test, y_test):
        start_time = time.time()
        clfs = {
            'lr': LogisticRegression(random_state=7),
            'dt': DecisionTreeClassifier(max_depth=2),
            'rf': RandomForestClassifier(n_estimators=2, random_state=7),
            'sv': SVC(kernel="linear", C=0.025, probability=True),
            # 'qd': QuadraticDiscriminantAnalysis(),
            'kn': KNeighborsClassifier(n_neighbors=15)
        }
        custom_ensemble = VotingClassifier([('clf1', clfs.get('lr')),
                                            ('clf2', clfs.get('dt')),
                                            # ('clf3', clfs.get('sv')),
                                            # ('clf3', clfs.get('qd')),
                                            ('clf4', clfs.get('kn')),
                                            ('clf5', clfs.get('rf'))], voting='soft')
        clfs['ce'] = custom_ensemble
        # train classifiers
        clfs_trained = {
            'lr': clfs.get('lr').fit(x_train, y_train),
            'dt': clfs.get('dt').fit(x_train, y_train),
            'rf': clfs.get('rf').fit(x_train, y_train),
            'sv': clfs.get('sv').fit(x_train, y_train),
            # 'qd': clfs.get('qd').fit(x_train, y_train),
            'kn': clfs.get('kn').fit(x_train, y_train),
            'ce': clfs.get('ce').fit(x_train, y_train)
        }
        for idx, (model_name, model) in enumerate(clfs_trained.items()):
            simple_results = cross_val_score(model, x_test, y_test, cv=3, scoring='accuracy')
            print('trained model: ', model_name, ' accuracy: ', simple_results.mean() * 100.0)
        # trained and named # TODO Auto create of this dictionary
        clfs_rdy = {
            "Logistic Regression": clfs_trained.get('lr'),
            "Decision Tree": clfs_trained.get('dt'),
            "Random Forest": clfs_trained.get('rf'),
            "Support Vector Classifier": clfs_trained.get('sv'),
            # "Quadratic Discriminant Analysis": clfs_trained.get('qd'),
            "K Neighbors Classifier": clfs_trained.get('kn'),
            "Custom Ensemble": clfs_trained.get('ce')
        }
        # print('tree: ', clfs_trained.get('dt').tree_.max_depth)
        end_time = time.time()
        print("Total training time: {} seconds".format(round(float(end_time - start_time), 2)))
        return clfs_rdy
