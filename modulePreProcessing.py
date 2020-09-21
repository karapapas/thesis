import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from tsfresh import extract_features
from tsfresh import select_features

class ScalingMethods:
    
    def useMinMax(self, dataset, ignoredFields):
        '''
        TODO να γίνεται το hstack με τα index των ignored αυτόματα
        '''
        # minMaxScaler = MinMaxScaler(feature_range=(0.1, 0.9))
        
        # columnsToScale = pd.Series(dataset.columns.array).values.tolist()
        # for i in ignoredFields:
        #     try:
        #         columnsToScale.remove(i)
        #     except:
        #         pass
        
        # # I will normalize every feature, including target class, except ignored fields
        # scaledMatrix = minMaxScaler.fit_transform(dataset[columnsToScale].values)
 
        # # concatenate session id and session starting time before recreating dataframe
        # scaledMatrixComplete = np.hstack((dataset.iloc[:,0:2].values, scaledMatrix))
        
        # # recreate dataframe
        # dataset = pd.DataFrame(scaledMatrixComplete, index=dataset.index, columns=dataset.columns)
        # return dataset
    
class FeatureEngineeringAndSelectionMethods:
    
    '''
    Επιστρέφει ένα νέο dataframe 
    μόνο με τα features που έχουν άμεση σχέση με τα features ενός round
    συν τα gsId και gsTime που είναι απαραίτητα για time series ανάλυση
    '''
    def getDatasetForTsAnalysis(self, dataset):
        datasetForTsAnalysis = dataset[['gsId',
                                        'gsTime',
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
        datasetWithTsExtractedFeatures = extract_features(datasetForTsAnalysis, column_id='gsId', column_sort='gsTime')
        
        # drop 100% nan features
        # newDsNoNan = newDs.dropna(axis=1, how='all')
        datasetWithTsExtractedFeaturesNoNan = datasetWithTsExtractedFeatures.dropna(axis=1, how='all')
        
        return datasetWithTsExtractedFeaturesNoNan

    def selectFeaturesUsingTsFresh(self, df):
        df = df.set_index('gsId')
        # auto feature selection. h target class epistrefei sth thesi 0.
        newDf = select_features(df, df.iloc[:,df.columns.get_loc('target_class')], fdr_level=0.05)
        return newDf
    
    def removeLowVarianceFeatures(self, df, thresholdValue):
        # Αφαιρώ manually το target class γιατί δεν θέλω να εμπλακεί στην αφαίρεση στηλών με βάση το variance
        selectedFsNoTargetClass = df.drop(['target_class'], axis=1)
        
        # function that calculates threshold th = (.9 * (1 - .9))
        selector = VarianceThreshold(threshold=thresholdValue)
        selectedFeaturesMatrix = selector.fit_transform(selectedFsNoTargetClass)
        
        # Query selector για να πάρω τα indices από τα επιλεγμένα features για να μπορέσω να κατασκευάσω ξανά dataframe
        selectedFeaturesIndices = selector.get_support(indices=True)
        
        # Ανακατασκευάζω το dataframe
        dfToReturn = pd.DataFrame(selectedFeaturesMatrix, index=selectedFsNoTargetClass.index, columns=selectedFsNoTargetClass.iloc[:,selectedFeaturesIndices].columns)
        
        # Προσθέτω ξανά το target class
        dfToReturn = pd.concat([dfToReturn, df[['target_class']]], axis=1, ignore_index=False)
        return dfToReturn