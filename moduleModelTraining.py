from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class TrainingMethods:

    # takes a dataframe and a list of features to train on
    # returns a dictionary of trained models
    @staticmethod
    def train_models(x_train, y_train, x_test, y_test):
        start_time = time.time()
        clfs = {
            'lr': LogisticRegression(random_state=7),
            'dt': DecisionTreeClassifier(random_state=7),
            'rf': RandomForestClassifier(n_estimators=100, random_state=7),
            'sv': SVC(kernel="rbf", C=1, probability=True),
            'gn': GaussianNB(),
            'mp': MLPClassifier(random_state=7, max_iter=200),
            'kn': KNeighborsClassifier(n_neighbors=5)
        }
        custom_ensemble = VotingClassifier([('clf1', clfs.get('lr')),
                                            ('clf2', clfs.get('dt')),
                                            ('clf3', clfs.get('gn')),
                                            # ('clf4', clfs.get('gn_o')),
                                            ('clf5', clfs.get('kn')),
                                            ('clf6', clfs.get('rf'))], voting='soft')
        clfs['ce'] = custom_ensemble
        # train classifiers
        clfs_trained = {
            'lr': clfs.get('lr').fit(x_train, y_train),
            'dt': clfs.get('dt').fit(x_train, y_train),
            'rf': clfs.get('rf').fit(x_train, y_train),
            'sv': clfs.get('sv').fit(x_train, y_train),
            'gn': clfs.get('gn').fit(x_train, y_train),
            'mp': clfs.get('mp').fit(x_train, y_train),
            'kn': clfs.get('kn').fit(x_train, y_train),
            'ce': clfs.get('ce').fit(x_train, y_train)
        }
        for idx, (model_name, model) in enumerate(clfs_trained.items()):
            simple_results = cross_val_score(model, x_test, y_test, cv=3, scoring='accuracy')
            print('Trained model: ', model_name, ' accuracy: ', (simple_results.mean() * 100.0).round(2))
        clfs_rdy = {
            "Logistic Regression": clfs_trained.get('lr'),
            "Decision Tree": clfs_trained.get('dt'),
            "Random Forest": clfs_trained.get('rf'),
            "Support Vector Classifier": clfs_trained.get('sv'),
            "Gaussian Naive Bayes": clfs_trained.get('gn'),
            "Multi-layer Perceptron": clfs_trained.get('mp'),
            "K Neighbors Classifier": clfs_trained.get('kn'),
            "Custom Ensemble": clfs_trained.get('ce')
        }
        end_time = time.time()
        print("Total training time: {} seconds".format(round(float(end_time - start_time), 2)))
        return clfs_rdy

    @staticmethod
    def train_models_fs_manual(x_train, y_train, x_test, y_test):
        start_time = time.time()
        clfs = {
            # optimized for manually selected features
            'lr_o': LogisticRegression(random_state=7, penalty='none', solver='newton-cg'),
            'dt_o': DecisionTreeClassifier(random_state=7, criterion='gini', max_depth=7),
            'rf_o': RandomForestClassifier(n_estimators=13, random_state=7, criterion='gini', max_depth=5),
            'sv_o': SVC(random_state=7, kernel="linear", C=2.5, probability=True, degree=1, gamma='scale'),
            'gn_o': GaussianNB(var_smoothing=0.0),
            'mp_o': MLPClassifier(random_state=7, max_iter=200, activation='relu', solver='lbfgs'),
            'kn_o': KNeighborsClassifier(n_neighbors=3, algorithm='auto', weights='uniform')
        }
        custom_ensemble = VotingClassifier([('clf1', clfs.get('lr_o')),
                                            ('clf2', clfs.get('dt_o')),
                                            ('clf3', clfs.get('gn_o')),
                                            ('clf5', clfs.get('kn_o')),
                                            ('clf6', clfs.get('rf_o'))], voting='soft')
        clfs['ce_o'] = custom_ensemble
        clfs_trained = {
            'lr_o': clfs.get('lr_o').fit(x_train, y_train),
            'dt_o': clfs.get('dt_o').fit(x_train, y_train),
            'rf_o': clfs.get('rf_o').fit(x_train, y_train),
            'sv_o': clfs.get('sv_o').fit(x_train, y_train),
            'gn_o': clfs.get('gn_o').fit(x_train, y_train),
            'mp_o': clfs.get('mp_o').fit(x_train, y_train),
            'kn_o': clfs.get('kn_o').fit(x_train, y_train),
            'ce_o': clfs.get('ce_o').fit(x_train, y_train)
        }
        for idx, (model_name, model) in enumerate(clfs_trained.items()):
            simple_results = cross_val_score(model, x_test, y_test, cv=3, scoring='accuracy')
            print('Trained model: ', model_name, ' accuracy: ', (simple_results.mean() * 100.0).round(2))

        clfs_rdy = {
            "Logistic Regression Optimized": clfs_trained.get('lr_o'),
            "Decision Tree Optimized": clfs_trained.get('dt_o'),
            "Random Forest Optimized": clfs_trained.get('rf_o'),
            "Support Vector Classifier Optimized": clfs_trained.get('sv_o'),
            "Gaussian Naive Bayes Optimized": clfs_trained.get('gn_o'),
            "Multi-layer Perceptron Optimized": clfs_trained.get('mp_o'),
            "K Neighbors Classifier Optimized": clfs_trained.get('kn_o'),
            "Custom Ensemble with Optimized CLFs": clfs_trained.get('ce_o')
        }
        end_time = time.time()
        print("Total training time: {} seconds".format(round(float(end_time - start_time), 2)))
        return clfs_rdy

    @staticmethod
    def train_models_fs_auto(x_train, y_train, x_test, y_test):
        start_time = time.time()
        clfs = {
            # optimized for selectKBest chi2 selected features
            'lr_o': LogisticRegression(random_state=7, penalty='l1', solver='liblinear'),
            'dt_o': DecisionTreeClassifier(random_state=7, criterion='entropy', max_depth=5),
            'rf_o': RandomForestClassifier(n_estimators=10, random_state=7, criterion='entropy', max_depth=3),
            'sv_o': SVC(random_state=7, kernel="linear", C=0.5, probability=True, degree=1),
            'gn_o': GaussianNB(var_smoothing=0.0),
            'mp_o': MLPClassifier(random_state=7, max_iter=200, activation='identity', solver='lbfgs'),
            'kn_o': KNeighborsClassifier(n_neighbors=1, algorithm='auto', weights='uniform')
        }
        custom_ensemble = VotingClassifier([('clf1', clfs.get('lr_o')),
                                            ('clf2', clfs.get('dt_o')),
                                            ('clf3', clfs.get('gn_o')),
                                            ('clf5', clfs.get('kn_o')),
                                            ('clf6', clfs.get('rf_o'))], voting='soft')
        clfs['ce_o'] = custom_ensemble
        clfs_trained = {
            'lr_o': clfs.get('lr_o').fit(x_train, y_train),
            'dt_o': clfs.get('dt_o').fit(x_train, y_train),
            'rf_o': clfs.get('rf_o').fit(x_train, y_train),
            'sv_o': clfs.get('sv_o').fit(x_train, y_train),
            'gn_o': clfs.get('gn_o').fit(x_train, y_train),
            'mp_o': clfs.get('mp_o').fit(x_train, y_train),
            'kn_o': clfs.get('kn_o').fit(x_train, y_train),
            'ce_o': clfs.get('ce_o').fit(x_train, y_train)
        }
        for idx, (model_name, model) in enumerate(clfs_trained.items()):
            simple_results = cross_val_score(model, x_test, y_test, cv=3, scoring='accuracy')
            print('Trained model: ', model_name, ' accuracy: ', (simple_results.mean() * 100.0).round(2))
        clfs_rdy = {
            "Logistic Regression Optimized": clfs_trained.get('lr_o'),
            "Decision Tree Optimized": clfs_trained.get('dt_o'),
            "Random Forest Optimized": clfs_trained.get('rf_o'),
            "Support Vector Classifier Optimized": clfs_trained.get('sv_o'),
            "Gaussian Naive Bayes Optimized": clfs_trained.get('gn_o'),
            "Multi-layer Perceptron Optimized": clfs_trained.get('mp_o'),
            "K Neighbors Classifier Optimized": clfs_trained.get('kn_o'),
            "Custom Ensemble with Optimized CLFs": clfs_trained.get('ce_o')
        }
        end_time = time.time()
        print("Total training time: {} seconds".format(round(float(end_time - start_time), 2)))
        return clfs_rdy
