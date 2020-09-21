import pandas as pd
from moduleLoading import LoadingMethods as ldr

dataset = ldr.connectAndFetch("127.0.0.1", "mci_db", "root", "toor", "SELECT * FROM v7")

targetClassIndex = dataset.columns.get_loc("mmse_pre_init")

# info
# datasetDesc = dataset.describe()
# datasetCorr = dataset.corr()

# from sklearn.preprocessing import StandardScaler
# stdScaler = StandardScaler()
# ds = dataset.values
# ds = stdScaler.fit_transform(ds)

# get position of target class !target class should be last in the view
# tc = dataset.columns.get_loc(dataset.columns[-1])

# select columns to include in tsfresh auto feature extraction process
# tsfresh mono gia features numerical. prosoxh oxi gia categorical pou exw kanei transform se number representation.
datasetForTs = dataset[['gsId', 'gsTime', 'total_rounds_in_session', 'total_success_rounds_in_session', 'total_success_round_points_in_session', 'avg_round_time_in_session','avg_round_time_for_souccess_rounds_in_session', 'mmse_pre_init']]

# για να αποφύγω τα NAN που θα προκύψουν κατά το feature extraction, λόγω των μηδενικών τιμών, όπως είναι οι 0 κερδισμένοι γύροι σε ένα session,
# σε καμία περίπτωση δεν θέλω να χάσω τη πληροφορία, οπότε οι επιλογές μου είναι περιορισμένες
# δεν έχω nan για να κάνω κάποιου είδους imputation
# και δεν θα είχε πολύ νόημα να αυξήσω τα πάντα κατά 1 ή κατά 0.1 για τα features τύπου float
# οπότε, μιας και σίγουρα θα χρειατεί να εφαρμόσω normilization, επιλέγω να γίνει σε αυτό το στάδιο και με range που δεν θα περιλαμβάνει το 0.
from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler(feature_range=(0.1, 0.9))
# dsScaledMatrix = minMaxScaler.fit_transform(datasetForTs)
dsScaledMatrix = minMaxScaler.fit_transform(datasetForTs[['total_rounds_in_session', 'total_success_rounds_in_session', 'total_success_round_points_in_session', 'avg_round_time_in_session','avg_round_time_for_souccess_rounds_in_session', 'mmse_pre_init']].values)

import numpy as np
fullMatrix = np.hstack((datasetForTs.iloc[:,0:2].values, dsScaledMatrix))
# datasetForTsScaled = pd.DataFrame(dsScaledMatrix, columns=['total_rounds_in_session', 'total_success_rounds_in_session', 'total_success_round_points_in_session', 'avg_round_time_in_session','avg_round_time_for_souccess_rounds_in_session', 'mmse_pre'])
# επειδή ο mixmaxscaler όπως όλοι οι scalers της βιβλιοθήκης scikit  επιστρέφει array (of arrays) για τη μετατροπή των αποτελεσμάτων ξανά σε df:
# datasetForTsScaled = pd.DataFrame(dsScaledArray, index=datasetForTs.iloc[:, 2:].index, columns=datasetForTs.iloc[:, 2:].columns)
datasetForTsScaled = pd.DataFrame(fullMatrix, index=datasetForTs.index, columns=datasetForTs.columns)

from tsfresh import extract_features
from tsfresh import select_features
from sklearn.feature_selection import VarianceThreshold

xf = extract_features(datasetForTsScaled, column_id='gsId', column_sort='gsTime')

# ta original features (demographics, medical, etc..) alla oute kai ta noumerical ('avg_round_time_in_session' in xx.columns == false) 
# den yparxoun sto dataframe pou epistrefei h extract_features. opote kanw join to arxiko dataframe
alldata = xf.join(dataset.set_index('gsId'))
# alldata = xf.join(dataset)

# drop tis stiles pou einai mono nan
xfNoNAN = alldata.dropna(axis=1, how='all')

# diagrafh olwn twn features me minor variance
selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
# selector = VarianceThreshold()
matrixWithoutZeroVarianceColumns = selector.fit_transform(xfNoNAN)
# get integer index, of the features selected
indicesOfSelectedFeatures = selector.get_support(indices=True)
#recreate dataframe after removing low variance features
newDf = pd.DataFrame(matrixWithoutZeroVarianceColumns, index=xfNoNAN.index, columns=xfNoNAN.iloc[:,indicesOfSelectedFeatures].columns)

# auto feature selection. h target class epistrefei sth thesi 0.
selectedXf = select_features(newDf, newDf.iloc[:,newDf.columns.get_loc(newDf.columns[-1])], fdr_level=0.05)

# entries of the independent features
# X = dataset.iloc[:, :tc].values #iloc for target class on the end
X = selectedXf.iloc[:,1:].values
# sX = ds[:,:tc]

# entries of the target class
y = selectedXf.iloc[:, :1].values
# sY = ds[:, tc]

# Για απλές περιπτώσεις που θέλω να ελέγξω το αποτέλεσμα για ένα split, διαφορετικά για εκπαίδευση μοντέλου θέλω cross-validation
# Splitting the dataset into the Training set and Test set (20% of our data) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=654)

# define the models
# dtModel = DecisionTreeRegressor(random_state=700)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
lrModel = LinearRegression()
clModel = LogisticRegression(solver='liblinear')

# fit the models
# dtModel.fit(X_train, y_train)
lrModel.fit(X_train, y_train)
clModel.fit(X_train, y_train)

# save the model to disk
# import pickle
# filename = 'modelsClassification/finalized_model.sav'
# pickle.dump(lrModel, open(filename, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)

# lrResults = lrModel.score(X_test, Y_test)
# print("Accuracy on test set: %.3f%% (%.3f%%)" % (result.mean()*100.0, result.std()*100.0))

# clResults = clModel.score(X_test, Y_test)
    
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# kfold = KFold(n_splits=10, random_state=7)
# # model = LogisticRegression(solver='liblinear')
# lrKfoldResults = cross_val_score(lrModel, X, y, cv=kfold)
# print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

# clKfolddResults = cross_val_score(clModel, X, y, cv=kfold)
# print("Accuracy: %.3f%% (%.3f%%)" % (clKfolddResults.mean()*100.0, clKfolddResults.std()*100.0))

# targetClassIndex = dataset.columns.get_loc('target_class')
