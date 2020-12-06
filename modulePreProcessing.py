import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler, StandardScaler
from tsfresh import extract_features, select_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split


# functions
def boxplot_features(dataframe, title, height):
    except_columns = ['gsId', 'userId', 'gsStartTime', 'target_class']
    print_columns = [x for x in dataframe.columns if x not in except_columns]
    f, ax = plt.subplots(figsize=(18, height))
    sns.boxplot(data=dataframe[print_columns], orient="h", palette="Set3", showmeans=True).set_title(title)
    sns.stripplot(data=dataframe[print_columns], orient="h", size=4, color=".3", linewidth=0)
    plt.show()


user_specific_features = ['sex', 'education', 'laptop_usage', 'smartphone_usage', 'family_med_history',
                                 'exercising', 'marital_status_1', 'marital_status_3', 'hypertension']
session_specific_features = ['total_gr_in_gs', 'total_success_rounds_in_session', 'total_win_gr_points_in_gs',
                             'avg_gr_time_in_gs', 'avg_gr_time_win_gr_in_gs']


class TransformationMethods:

    @staticmethod
    def handle_outliers(df):
        df_beforehand = df.copy()

        for feature in df.columns:

            # quartiles are standard on 25th, 50th and 75% percentile
            q1_v = np.percentile(df[feature], 25).round(2)
            median_v = np.percentile(df[feature], 50).round(2)
            q3_v = np.percentile(df[feature], 75).round(2)
            iqr_v = (q3_v - q1_v).round(2)

            # min, max depends
            min_v = (q1_v - 1.5*iqr_v).round(2)
            max_v = (q3_v + 1.5*iqr_v).round(2)
            # print('feature ', feature, ' min=', min_v, ' Q1=', q1_v, ' Q2=', median_v, ' Q3', q3_v, ' max=', max_v)

            features_to_ignore = ['userId ', 'gsId ', 'gsStartTime', 'target_class', 'sex', 'education']
            if feature in features_to_ignore:
                pass
            elif re.search('(^total_)', feature):
                df.loc[(df[feature] > max_v), feature] = q3_v
                df.loc[(df[feature] < min_v), feature] = q1_v
            elif re.search('(^avg_)', feature):
                df.loc[(df[feature] < min_v), feature] = median_v
                df.loc[(df[feature] > max_v), feature] = median_v
            else:
                df.loc[(df[feature] < min_v), feature] = q1_v
                df.loc[(df[feature] > max_v), feature] = q3_v

        boxplot_features(df_beforehand[user_specific_features], 'User Specific Features. Before. ', 5)
        boxplot_features(df[user_specific_features], 'User Specific Features. After.', 5)
        boxplot_features(df_beforehand[session_specific_features], 'Session Specific Features. Before.', 5)
        boxplot_features(df[session_specific_features], 'Session Specific Features. After.', 5)
        return df

    @staticmethod
    def use_min_max(df, columnsToIgnoreList):
        boxplot_features(df, 'Before Scaling', 10)
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
        boxplot_features(dfAfterScaling, 'After Scaling', 10)
        return dfAfterScaling

    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    @staticmethod
    def use_standard_scaler(df, columns_to_ignore_list):
        boxplot_features(df, 'Before Scaling', 10)
        all_columns_list = pd.Series(df.columns.array).values.tolist()
        columns_to_scale_list = [x for x in all_columns_list if x not in columns_to_ignore_list]

        # partition dataframe
        df_part_to_scale = df[columns_to_scale_list]
        df_part_to_ignore = df[columns_to_ignore_list]

        # instantiate scaler
        standard_scaler = StandardScaler()

        # run scaler on the df part we want to scale
        scaled_matrix = standard_scaler.fit_transform(df_part_to_scale.values)

        # recreate df
        df_scaled_part = pd.DataFrame(scaled_matrix, index=df_part_to_scale.index, columns=df_part_to_scale.columns)

        # concat ignored columns and scaled columns to one df
        df_after_scaling = pd.concat([df_part_to_ignore, df_scaled_part], axis=1, ignore_index=False)
        boxplot_features(df_after_scaling, 'After Scaling', 10)
        return df_after_scaling


class FeatureMethods:

    # example https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
    # Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
    @staticmethod
    def remove_low_variance_features(df, threshold_val, ddof_val):
        try:
            # exclude target, time and id from the process
            features_to_cal_variance_for = df.drop(['target_class', 'userId', 'gsId', 'gsStartTime'], axis=1)
            variance = features_to_cal_variance_for.var(ddof=ddof_val).sort_values(ascending=False).round(10)
            f, ax = plt.subplots(figsize=(18, 10))
            sns.barplot(variance.values, variance.index, palette="Set3")
            plt.show()
            display(HTML(variance.to_frame().to_html()))
            all_columns = df.columns

            selector = VarianceThreshold(threshold=threshold_val)
            selected_features_matrix = selector.fit_transform(features_to_cal_variance_for)

            # query selector for the indices of the selected features to rebuild dataframe
            indices_of_feature_to_remain = selector.get_support(indices=True)

            # rebuild dataframe
            df_new = pd.DataFrame(selected_features_matrix, index=features_to_cal_variance_for.index,
                                  columns=features_to_cal_variance_for.iloc[:, indices_of_feature_to_remain].columns)

            # dfToReturn = pd.concat([dfToReturn, df[['target_class']]], axis=1, ignore_index=False)
            df_to_return = df_new.join(df[['target_class', 'userId', 'gsId', 'gsStartTime']], how='inner')

            features_to_keep = df_to_return.columns
            features_ruled_out = [x for x in all_columns if x not in features_to_keep]
            print("Threshold value: ", round(float(threshold_val), 2))
            print("Features ruled out: \n", features_ruled_out)
            return df_to_return
        except ValueError as e:
            print('ValueError exception:', e)

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

        # split dataframe samples for the evaluation inspection
        x = df[features]
        y = df.iloc[:, df.columns.get_loc('target_class')]
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=7)

        clf = RandomForestClassifier(max_depth=3, n_estimators=5, random_state=7)
        clf.fit(x_train, y_train)

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

        result = permutation_importance(clf, x_test, y_test, n_repeats=10, random_state=7)
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

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
        ax1.set_title("Spearman correlation coefficient (SciPy)")
        corr = spearmanr(x).correlation
        # print('corr=', corr)
        corr_linkage = hierarchy.ward(corr)
        feature_names = x.columns.tolist()
        dendro = hierarchy.dendrogram(corr_linkage, labels=feature_names, ax=ax1, leaf_rotation=0, orientation='right')
        dendro_idx = np.arange(0, len(dendro['ivl']))

        # correlation heatmap
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
        ax2.set_title("Compute pairwise correlation of columns (Pandas)")
        sns.heatmap(x.corr(method='pearson'), annot=True, linewidths=.4, fmt='.1f', ax=ax2)

        # show
        fig.tight_layout()
        plt.show()

    # example
    # https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html
    # https://nbviewer.jupyter.org/github/justmarkham/scikit-learn-tips/blob/master/notebooks/23_linear_model_coefficients.ipynb
    @staticmethod
    def inspection_using_regressors(df, features):

        # split dataframe samples for the evaluation inspection
        x = df[features]
        y = df.iloc[:, df.columns.get_loc('target_class')]
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=7)

        # SelectKBest is a wrapper. The default scorer algorithm is f_classif (ANOVA F-value between label/feature)
        # The smaller the p value the more significant the feature is, so we reverse its result for the plot
        skb = SelectKBest(chi2, k=2)
        skb_results = skb.fit(x_train, y_train)
        p_values = skb_results.pvalues_
        p_values = -np.log10(p_values)
        p_values /= p_values.max()
        f_scores = skb_results.scores_

        # Check coefficients of Linear Regression
        lr = LinearRegression().fit(x_train, y_train)
        lr.score(x_test, y_test)
        lr_coef_abs = np.abs(lr.coef_)

        # Check coefficients of Ridge Regression
        rr = Ridge(alpha=1.0).fit(x_train, y_train)
        rr.score(x_test, y_test)
        rr_coef_abs = np.abs(rr.coef_)

        # Check coefficients of Lasso Regression
        ls = Lasso(alpha=0.1).fit(x_train, y_train)
        ls.score(x_test, y_test)
        ls_coef_abs = np.abs(ls.coef_)

        # Check coefficients of ElasticNet
        en = ElasticNet(random_state=0).fit(x_train, y_train)
        en.score(x_test, y_test)
        en_coef_abs = np.abs(en.coef_)

        df_to_plot = pd.DataFrame({'SelectKBest F scores': f_scores,
                                   'SelectKBest P values ($-Log(p_{value})$)': p_values,
                                   'Linear Regression coef.(abs)': lr_coef_abs,
                                   'Ridge Regression coef.(abs)': rr_coef_abs,
                                   'Lasso Regression coef.(abs)': ls_coef_abs,
                                   'ElasticNet coef.(abs)': en_coef_abs
                                   }, index=features)

        plt.figure(1)
        plt.clf()
        ax = df_to_plot.plot.barh().legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.show()

        # get selected as: array([3, 4], dtype=int64)
        selected_features_indices = skb.get_support(indices=True)

        # get selected as list: ['age', 'education']
        # return x.columns[selected_features_indices.tolist()].values.tolist()

    @staticmethod
    def one_hot_encode_features(df, features):
        ohe = OneHotEncoder(sparse=False)
        df_temp = df.copy()
        ohe.fit(df_temp[[features]])
        print('df shape before drop:', df.shape)
        df = df.drop([features], axis=1)
        print('df shape after drop, before ohe:', df.shape)
        df[ohe.get_feature_names([features])] = ohe.transform(df_temp[[features]])
        print('df shape after ohe:', df.shape)
        return df

    @staticmethod
    def discretize_features(df, features):
        df_before_discretize = df.copy()
        discretizer = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='quantile')
        df_part_to_disc = df[features]
        df[features] = discretizer.fit_transform(df_part_to_disc)
        # convert discretized columns from float to int
        df[features] = df[features].astype(int)
        return df

    '''
    Επιστρέφει μόνο τα extracted features
    Δεν επιστρέφει τα input features
    Δεν επιστρέφει column_id και column_sort
    '''
    def extractFeaturesUsingTsFresh(self, datasetForTsAnalysis):
        datasetWithTsExtractedFeatures = extract_features(datasetForTsAnalysis, column_id='gsId',
                                                          column_sort='gsStartTime')

        # drop 100% nan features
        # newDsNoNan = newDs.dropna(axis=1, how='all')
        datasetWithTsExtractedFeaturesNoNan = datasetWithTsExtractedFeatures.dropna(axis=1, how='all')

        return datasetWithTsExtractedFeaturesNoNan

    def selectFeaturesUsingTsFresh(self, df):
        df = df.set_index('gsId')
        # auto feature selection. h target class epistrefei sth thesi 0.
        newDf = select_features(df, df.iloc[:, df.columns.get_loc('target_class')], fdr_level=0.05)
        return newDf
