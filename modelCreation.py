# imports
from moduleLoading import LoadingMethods
from modulePreProcessing import ScalingMethods, FeatureMethods
from moduleModelTraining import TrainingMethods
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from numpy import where
from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix
from sklearn.feature_selection import VarianceThreshold

# class instances
load = LoadingMethods()
scale = ScalingMethods()
features = FeatureMethods()
train = TrainingMethods()

# connect to db and fetch data
df = load.connect_and_fetch("127.0.0.1", "mci_db", "root", "toor", "SELECT * FROM v7")
# bxplot = df.boxplot(column=["total_rounds_in_session"])
# plt.show()


# define target class
df = load.separate_target_class(df, "moca_pre_binary_binned")

# define features by type and encoding method
# categorical for one-hot
f_cat = ['sex', 'marital_status']
# ordinal continuous numerical for binning
f_ord = ['age', 'total_gr_in_gs', 'total_win_gr_points_in_gs',
         'avg_gr_time_in_gs', 'avg_gr_time_win_gr_in_gs']
# ordinal numerical already binned
f_num = ['education', 'laptop_usage', 'smartphone_usage', 'smoking',
         'family_med_history', 'exercising', 'depression', 'hypertension']

all_f = f_cat + f_ord + f_num

X = df[f_cat + f_ord + f_num]
y = df.iloc[:, df.columns.get_loc('target_class')].values

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# for label, _ in counter.items():
#     row_ix = where(y == label)[0]
#     pyplot.scatter(X.iloc[row_ix, 4], X.iloc[row_ix, 11], label=str(label))
# pyplot.legend()
# pyplot.show()
#
# for label, _ in counter.items():
#     row_ix = where(y == label)[0]
#     pyplot.scatter(X.iloc[row_ix, 13:14], y[row_ix], label=str(label))
# pyplot.legend()
# pyplot.show()

# X = select.removeBasedOnVariance(X, 0.15)
# X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=7)

categorical_pipe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('scale', StandardScaler(with_mean=False))
])
ordinal_pipe = Pipeline([
    ('bin', KBinsDiscretizer(n_bins=5, encode='ordinal')),
    ('scale', StandardScaler(with_mean=False))
])
numerical_pipe = Pipeline([
    ('scale', StandardScaler(with_mean=False))])

preprocessingEncoding = ColumnTransformer([
    ('cat', categorical_pipe, f_cat),
    ('bin', ordinal_pipe, f_ord),
    ('num', numerical_pipe, f_num)
])

pl = Pipeline([
    ('enc', preprocessingEncoding),
    # ('classifier', RandomForestClassifier(random_state=7))
    ('classifier', LogisticRegression(random_state=7))
])

pl.fit(X_train, y_train)
pl.score(X_test, y_test)
model = pl['classifier']
disp = plot_confusion_matrix(model, X_test, y_test, cmap='Blues', values_format='d')



