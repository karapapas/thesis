import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from moduleMetrics import MetricsMethods
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import plot_roc_curve

metricsCi = MetricsMethods()

class TrainingMethods:
    
    def trainLinearRegressionModel(self, df, metrics):
        # entries of the target class
        targetClassIndex = df.columns.get_loc('target_class')
        y = df.iloc[:, targetClassIndex].values
        
        # entries of the independent features
        allColumns = pd.Series(df.columns.array).values.tolist()
        try:
            allColumns.remove('target_class')
            allColumns.remove('gsTime')
        except:
            pass
        X = df[allColumns]
        
        # Manual split για εκπαίδευση μοντέλου
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=654)
        
        # define the models
        model = LinearRegression()
        
        # fit the models
        model.fit(X_train, y_train)
        
        # Αποτελέσματα για το split στο οποίο βασίστικε το traininig. ;Άνευ σημασίας.
        # results = model.score(X_test, y_test)
        # print("Accuracy on test set: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

        kfold = KFold(n_splits=10, random_state=7)
        
        lrKfoldResults = cross_val_score(model, X, y, cv=kfold)
        print("Accuracy: %.3f%% (%.3f%%)" % (lrKfoldResults.mean()*100.0, lrKfoldResults.std()*100.0))

        metricsCi.generate_metrics(model, X, y, metrics)
        
        '''plots start'''
        # Make predictions using the testing set
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_values = X_train.values[:,0]
        y_pred = model.predict(X_test.values[:,5:6])
        self.y_pred = y_pred

        plt.scatter(X_test.values[:,5:6], y_test,  color='black')
        plt.plot(X_test.values[:,5:6], y_pred, color='blue', linewidth=3)

        plt.xticks(())
        plt.yticks(())

        plt.show()
        '''plots end'''
        
        return model
    
    def trainClassificationModel(self, df, mustIgnore, mustInclude, metrics):
        
        # target class entries
        targetClassIndex = df.columns.get_loc('target_class')
        y = df.iloc[:, targetClassIndex].values
        
        # entries of the independent features
        allColumns = pd.Series(df.columns.array).values.tolist()
        
        # make sure target class, gsTime and gsId is not part of X features
        try:
            for i in mustIgnore:
                if(i in allColumns):
                    allColumns.remove(i)
        except:
            pass
        X = df[allColumns]

        '''
        Επιπλέον feature selection
        selector = SelectKBest(chi2, k=4)
        X = selector.fit_transform(X, y)
        self.X_after = X
        self.selectedFeaturesTrue = allColumns[selector.get_support(indices=True)
        '''
        
        X_mandatory = df[mustInclude].values
        # self.toKeep = df[mustInclude].values
        
        X = np.append(X, X_mandatory, axis=1)
        # self.complete = X
        print(X.head())
        '''
        Manual split για εκπαίδευση μοντέλου
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        # self.sample = X_test
        
        # define the models
        model = LogisticRegression(random_state=7).fit(X_train, y_train)
        
        '''metrics'''
        # Αποτελέσματα για το split στο οποίο βασίστικε το traininig. Άνευ σημασίας.
        results = model.score(X_test, y_test)
        # print("Accuracy on test set: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
        kfold = KFold(n_splits=10, random_state=7)
        results = cross_val_score(model, X, y, cv=kfold)
        print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
        metricsCi.generate_metrics(model, X, y, metrics)
        
        plot_roc_curve(model, X_test, y_test)

        return model
        