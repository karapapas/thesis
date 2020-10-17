from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import time


class TrainingMethods:

    # takes a dataframe and a list of features to train on
    # returns a dictionary of trained models
    @staticmethod
    def train_models(x_train, y_train):
        start_time = time.time()
        clfs = {
            'lr': LogisticRegression(random_state=7),
            'dt': DecisionTreeClassifier(max_depth=3),
            'rf': RandomForestClassifier(max_depth=3, n_estimators=4, random_state=7),
            'sv': SVC()
        }
        custom_ensemble = VotingClassifier([('clf1', clfs.get('lr')),
                                            ('clf2', clfs.get('dt')),
                                            # ('clf3', clfs.get('sv')),
                                            ('clfe', clfs.get('rf'))], voting='soft')
        clfs['ce'] = custom_ensemble
        # train classifiers
        clfs_trained = {
            'lr': clfs.get('lr').fit(x_train, y_train),
            'dt': clfs.get('dt').fit(x_train, y_train),
            'rf': clfs.get('rf').fit(x_train, y_train),
            'sv': clfs.get('sv').fit(x_train, y_train),
            'ce': clfs.get('ce').fit(x_train, y_train)
        }
        # trained and named # TODO Auto create of this dictionary
        clfs_rdy = {
            "Logistic Regression": clfs_trained.get('lr'),
            "Decision Tree": clfs_trained.get('dt'),
            "Random Forest": clfs_trained.get('rf'),
            "Support Vector Classifier": clfs_trained.get('sv'),
            "Custom Ensemble": clfs_trained.get('ce')
        }
        end_time = time.time()
        print("Total training time: {} seconds".format(round(float(end_time - start_time), 2)))
        return clfs_rdy
