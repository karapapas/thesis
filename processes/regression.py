from moduleLoading import LoadingMethods
from modulePreProcessing import ScalingMethods
from modulePreProcessing import FeatureEngineeringAndSelectionMethods
from moduleModelTraining import TrainingMethods

ldCi = LoadingMethods()
scCi = ScalingMethods()
fsCi = FeatureEngineeringAndSelectionMethods()
tnCi = TrainingMethods()

df = ldCi.connectAndFetch("127.0.0.1", "mci_db", "root", "toor", "SELECT * FROM v7")
df = ldCi.separateTargetClass(df, "moca_pre_init")

'''
Κάνω scaling, βάζοντας range=(0.1, 0.9), πριν χρησιμοποιήσω την TSFresh,
με σκοπό να αποφύγω τις μηδενικές τιμές στα data μου,
oι οποίες θα οδηγούσαν σε NaN τιμές στα extracted features.
'''
df = scCi.useMinMax(df, ['gsId', 'gsTime'])

# select features to apply TS feature extraction
dfForTsAnalysis = fsCi.getDatasetForTsAnalysis(df)

# TS feature extraction
dfWithTsFeatures = fsCi.extractFeaturesUsingTsFresh(dfForTsAnalysis)

'''
examples https://pandas.pydata.org/docs/user_guide/merging.html#joining-key-columns-on-an-index
'''
dfComplete = df.join(dfWithTsFeatures, on='gsId')


'''
Κάνω ξανά scaling, αυτή τη φορά για τα features από το TsFresh
'''
dfAllScaled = scCi.useMinMax(dfComplete, ['gsId', 'gsTime'])


'''
TS Fresh auto feature_selection
Η target_class επιστρέφει στο index=0, Η gsTime στη τελευταία θέση, Η gsId έχει αφαιρεθεί
https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_selection.html#module-tsfresh.feature_selection.selection
!!! Εδώ βλέπω πως πολλά features είναι correlated οπότε θα ήταν ενα καλό σημείο για PCA
'''
dfSelectedFs = fsCi.selectFeaturesUsingTsFresh(dfAllScaled)
dfSelectedFs = dfSelectedFs.drop(['gsTime'], axis=1)


'''
Αφαίρεση των features που έχουν μικρό ή μηδενικό variance
Η διαδικασία γίνεται σε αυτό το σημείο γιατί εδώ υπάρχουν πλέον
όλα τα features συγκεντρωμένα από τις προηγούμενες διαδικασίες
'''
dfVarianceFiltered = fsCi.removeLowVarianceFeatures(dfSelectedFs, 0.07)


'''
Model Training
Από τα selected features και αφού έχω αφαιρέσει αυτά τα οποία έχουν μικρό variance
Θ΄΄ελω ένα dataframe το οποίο να έχει το target_class και να μην έχει το gsId και gsTime
Για να προχωρήσω με το splitting και την εκπαίδευση του μοντέλου
'''
model = tnCi.trainLinearRegressionModel(dfVarianceFiltered, ['r2', 'neg_mean_absolute_error'])
