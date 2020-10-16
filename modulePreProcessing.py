import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from tsfresh import extract_features, select_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from scipy.cluster import hierarchy


def boxplot_features(dataframe, title):
    except_columns = ['gsId', 'gsStartTime', 'target_class']
    print_columns = [x for x in dataframe.columns if x not in except_columns]
    sns.boxplot(data=dataframe[print_columns], orient="h", palette="Set3", showmeans=True).set_title(title)
    plt.show()


class ScalingMethods:

    @staticmethod
    def handle_outliers(df):
        boxplot_features(df, 'Before removing outliers')

        for feature in df.columns:
            if re.search('(^total_)', feature):
                max_val = np.percentile(df[feature], 99.25)
                q3_val = np.percentile(df[feature], 75)
                df.loc[(df[feature] > max_val), feature] = q3_val
            elif re.search('(^avg_)', feature):
                max_val = np.percentile(df[feature], 99.25)
                mean_val = df[feature].mean()
                df.loc[(df[feature] > max_val), feature] = mean_val

        boxplot_features(df, 'After removing outliers')
        return df

    @staticmethod
    def use_min_max(df, columnsToIgnoreList):
        boxplot_features(df, 'Before Scaling')
        allColumnsList = pd.Series(df.columns.array).values.tolist()
        columnsToScaleList = [x for x in allColumnsList if x not in columnsToIgnoreList]

        # partition dataframe
        dfPartToScale = df[columnsToScaleList]
        dfPartToIgnore = df[columnsToIgnoreList]

        # instantiate scaler
        minMaxScaler = MinMaxScaler(feature_range=(0.1, 0.9))

        # run scaler on the df part we want to scale
        scaledMatrix = minMaxScaler.fit_transform(dfPartToScale.values)

        # recreate df
        dfScaledPart = pd.DataFrame(scaledMatrix, index=dfPartToScale.index, columns=dfPartToScale.columns)

        # concat ignored columns and scaled columns to one df
        dfAfterScaling = pd.concat([dfPartToIgnore, dfScaledPart], axis=1, ignore_index=False)
        boxplot_features(dfAfterScaling, 'After Scaling')
        return dfAfterScaling


class FeatureMethods:

    @staticmethod
    def remove_low_variance_features(df, thresholdValue):

        # exclude target, time and id from the process
        selectedFsNoTargetClass = df.drop(['target_class', 'gsId', 'gsStartTime'], axis=1)
        variance = selectedFsNoTargetClass.var().sort_values(ascending=False)
        sns.barplot(variance.values, variance.index, palette="Set3")
        for i, v in variance.items():
            print(i, round(float(v), 1))

        dfColumnsToRuleOut = df.columns

        # function that calculates threshold th = (.9 * (1 - .9))
        selector = VarianceThreshold(threshold=thresholdValue)
        selectedFeaturesMatrix = selector.fit_transform(selectedFsNoTargetClass)

        # Query selector για να πάρω τα indices από τα επιλεγμένα features για να μπορέσω να κατασκευάσω ξανά dataframe
        selectedFeaturesIndices = selector.get_support(indices=True)

        # Ανακατασκευάζω το dataframe
        dfToReturn = pd.DataFrame(selectedFeaturesMatrix, index=selectedFsNoTargetClass.index, columns=selectedFsNoTargetClass.iloc[:, selectedFeaturesIndices].columns)

        # Προσθέτω ξανά το target class
        # dfToReturn = pd.concat([dfToReturn, df[['target_class']]], axis=1, ignore_index=False)
        dfToReturn = dfToReturn.join(df[['target_class', 'gsId', 'gsStartTime']], how='inner')

        dfColumnsFiltered = dfToReturn.columns
        featuresRuledOut = [x for x in dfColumnsToRuleOut if x not in dfColumnsFiltered]
        print("Threshold value: ", round(float(thresholdValue), 2))
        print("Features ruled out: \n", featuresRuledOut)
        return dfToReturn

    # takes a dataframe
    # plots feature importance (MDI) using Random Forest Classifier
    # plots feature permutation importance
    # based on examples:
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    # related papers:
    # Gilles Louppe, Understanding variable importance in forests of randomized trees
    # TODO include a random feature
    @staticmethod
    def inspection_using_classifier(df, features):

        # independent variables
        x = df[features]

        # target class
        target_class_index = df.columns.get_loc('target_class')
        y = df.iloc[:, target_class_index]

        # Split dataset to select feature and evaluate the classifier
        X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=7)

        clf = RandomForestClassifier(max_depth=3, n_estimators=5, random_state=7)
        clf.fit(X_train, y_train)
        # results = clf.score(X_test, y_test)
        # print("Accuracy on test set: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        feature_names = x.columns
        tree_importance_sorted_idx = np.argsort(clf.feature_importances_)

        y_ticks = x.columns.to_numpy()  # pandas.core.indexes.base.Index to numpy.ndarray
        tree_indices = np.arange(0, len(feature_names))

        ax1.set_title("Random Forest Feature Importance (MDI)")
        ax1.barh(y_ticks, clf.feature_importances_[tree_importance_sorted_idx], height=0.7, align='center')
        ax1.set_yticklabels(feature_names[tree_importance_sorted_idx])
        ax1.set_yticks(y_ticks)
        ax1.set_ylim((-0.5, len(clf.feature_importances_) - 0.5))

        result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=7)
        perm_sorted_idx = result.importances_mean.argsort()
        ax2.set_title("Permutation Importance")
        ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False, labels=feature_names[perm_sorted_idx])

        fig.tight_layout()
        plt.show()

    # takes a dataframe and a list of features to inspect
    # plots hierarchy of feature correlation "clusters" and correlation heatmap
    # example
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#handling-multicollinear-features
    @staticmethod
    def correlation_inspection(df, fs):

        x = df[fs]

        # hierarchy of feature correlation "clusters"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        corr = spearmanr(x).correlation
        corr_linkage = hierarchy.ward(corr)
        feature_names = x.columns.tolist()
        dendro = hierarchy.dendrogram(corr_linkage, labels=feature_names, ax=ax1, leaf_rotation=0, orientation='right')
        dendro_idx = np.arange(0, len(dendro['ivl']))

        # correlation heatmap
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
        sns.heatmap(x.corr(method='pearson'), annot=True, linewidths=.4, fmt='.1f', ax=ax2)

        # show
        fig.tight_layout()
        plt.show()


    def removeBasedOnVariance(self, df, thresholdValue):

        selector = VarianceThreshold(threshold=thresholdValue)
        selectedFeaturesMatrix = selector.fit_transform(df)
        # self.selectedFeaturesMatrix = selectedFeaturesMatrix

        # Query selector για να πάρω τα indices από τα επιλεγμένα features για να μπορέσω να κατασκευάσω ξανά dataframe
        selectedFeaturesIndices = selector.get_support(indices=True)

        # Ανακατασκευάζω το dataframe
        df = pd.DataFrame(selectedFeaturesMatrix, index=df.index,
                                  columns=df.iloc[:, selectedFeaturesIndices].columns)
        # self.dfToReturn = df
        return df

    '''
    Επιστρέφει ένα νέο dataframe 
    μόνο με τα features που έχουν άμεση σχέση με τα features ενός round
    συν τα gsId και gsStartTime που είναι απαραίτητα για time series ανάλυση
    '''
    def getDatasetForTsAnalysis(self, dataset):
        datasetForTsAnalysis = dataset[['gsId',
                                        'gsStartTime',
                                        'total_rounds_in_session',
                                        'total_success_rounds_in_session',
                                        'total_success_round_points_in_session',
                                        'avg_round_time_in_session',
                                        'avg_round_time_for_souccess_rounds_in_session',
                                        ]].copy()
        return datasetForTsAnalysis

    '''
    Επιστρέφει μόνο τα extracted features
    Δεν επιστρέφει τα input features
    Δεν επιστρέφει column_id και column_sort
    '''
    def extractFeaturesUsingTsFresh(self, datasetForTsAnalysis):
        datasetWithTsExtractedFeatures = extract_features(datasetForTsAnalysis, column_id='gsId', column_sort='gsStartTime')

        # drop 100% nan features
        # newDsNoNan = newDs.dropna(axis=1, how='all')
        datasetWithTsExtractedFeaturesNoNan = datasetWithTsExtractedFeatures.dropna(axis=1, how='all')

        return datasetWithTsExtractedFeaturesNoNan

    def selectFeaturesUsingTsFresh(self, df):
        df = df.set_index('gsId')
        # auto feature selection. h target class epistrefei sth thesi 0.
        newDf = select_features(df, df.iloc[:,df.columns.get_loc('target_class')], fdr_level=0.05)
        return newDf
