import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from tsfresh import extract_features
from tsfresh import select_features
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_features(dataframe, title):
    except_columns = ['gsId', 'gsStartTime', 'target_class']
    print_columns = [x for x in dataframe.columns if x not in except_columns]
    sns.boxplot(data=dataframe[print_columns], orient="h", palette="Set3", showmeans=True).set_title(title)
    plt.show()


class ScalingMethods:

    def handle_outliers(self, df):
        boxplot_features(df, 'Before removing outliers')

        for feature in df.columns:
            if re.search('(^total_)', feature):
                max_val = np.percentile(df[feature], 95)
                q3_val = np.percentile(df[feature], 75)
                df.loc[(df[feature] > max_val), feature] = q3_val
            elif re.search('(^avg_)', feature):
                max_val = np.percentile(df[feature], 95)
                mean_val = df[feature].mean()
                df.loc[(df[feature] > max_val), feature] = mean_val

        boxplot_features(df, 'After removing outliers')
        return df

    def useMinMax(self, df, columnsToIgnoreList):
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

class FeatureEngineeringMethods:

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

class FeatureSelectionMethods:

    def removeLowVarianceFeatures(self, df, thresholdValue):
        for i, v in df.var().items():
            print(i, round(float(v), 1))

        # Αφαιρώ manually το target class γιατί δεν θέλω να εμπλακεί στην αφαίρεση στηλών με βάση το variance
        selectedFsNoTargetClass = df.drop(['target_class', 'gsId', 'gsStartTime'], axis=1)
        # self.selectedFsNoTargetClass = selectedFsNoTargetClass
        dfColumnsToFilter = df.columns

        # function that calculates threshold th = (.9 * (1 - .9))
        selector = VarianceThreshold(threshold=thresholdValue)
        selectedFeaturesMatrix = selector.fit_transform(selectedFsNoTargetClass)
        # self.selectedFeaturesMatrix = selectedFeaturesMatrix

        # Query selector για να πάρω τα indices από τα επιλεγμένα features για να μπορέσω να κατασκευάσω ξανά dataframe
        selectedFeaturesIndices = selector.get_support(indices=True)

        # Ανακατασκευάζω το dataframe
        dfToReturn = pd.DataFrame(selectedFeaturesMatrix, index=selectedFsNoTargetClass.index, columns=selectedFsNoTargetClass.iloc[:,selectedFeaturesIndices].columns)
        # self.dfToReturn = dfToReturn
        # Προσθέτω ξανά το target class
        # dfToReturn = pd.concat([dfToReturn, df[['target_class']]], axis=1, ignore_index=False)
        dfToReturn = dfToReturn.join(df[['target_class', 'gsId', 'gsStartTime']], how='inner')

        dfColumnsFiltered = dfToReturn.columns
        featuresRuledOut = [x for x in dfColumnsToFilter if x not in dfColumnsFiltered]
        print("Threshold value ", thresholdValue)
        print("Features ruled out ", featuresRuledOut)
        return dfToReturn

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