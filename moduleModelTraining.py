import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from moduleMetrics import MetricsMethods as mm


class TrainingMethods:
    
    def trainLinearRegressionModel(df, metrics):
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

        mm.generateMetrics(model, X, y, metrics)

        return model
    
    def trainClassificationModel(df, metrics):
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
        model = LogisticRegression(random_state=0).fit(X_train, y_train)
        
        # Αποτελέσματα για το split στο οποίο βασίστικε το traininig. ;Άνευ σημασίας.
        # results = model.score(X_test, y_test)
        # print("Accuracy on test set: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

        kfold = KFold(n_splits=10, random_state=7)
        
        results = cross_val_score(model, X, y, cv=kfold)
        print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

        mm.generateMetrics(model, X, y, metrics)

        return model
        
