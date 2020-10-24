import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class UtilityMethods:

    # inspect the distribution of potential target classes
    @staticmethod
    def inspect_target_distribution(df):
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        plt.figtext(0.5, 0.95, 'Class Label Distribution of data set', ha="center", va="top", fontsize=16)

        dfl = df[['userId', 'mmse_pre_init', 'mmse_post_init', 'moca_pre_init', 'moca_post_init']]
        dfl.set_index('userId')
        dfl = dfl.groupby(by='userId').first()

        a_heights, a_bins = np.histogram(dfl['mmse_pre_init'])
        b_heights, b_bins = np.histogram(dfl['mmse_post_init'], bins=a_bins)
        c_heights, c_bins = np.histogram(dfl['moca_pre_init'])
        d_heights, d_bins = np.histogram(dfl['moca_post_init'], bins=c_bins)

        width_mmse = (a_bins[1] - a_bins[0])/3
        width_moca = (c_bins[1] - d_bins[0])/3

        axs[0, 0].set_title('MMSE Distribution')
        axs[0, 0].bar(a_bins[:-1], a_heights, width=width_mmse, facecolor='orange');
        axs[0, 0].bar(b_bins[:-1]+width_mmse, b_heights, width=width_mmse, facecolor='red');

        axs[0, 1].set_title('MOCA Distribution')
        axs[0, 1].bar(c_bins[:-1], c_heights, width=width_moca, facecolor='lightgreen');
        axs[0, 1].bar(d_bins[:-1]+width_moca, d_heights, width=width_moca, facecolor='green');

        # differentiation
        df_mmse = pd.DataFrame([['Standardised MMSE', 10, 11, 4, 5],
                               ['uiowa.edu (severe)', 0, 18, 6, 6],
                               ['uiowa.edu (cut-off)', 0, 0, 24, 6],
                               ['gov.gr', 12, 9, 4, 5]], columns=['Institution',
                                                                  'AD', 'AD-MCI', 'MCI',
                                                                  'No Cog. Imp.']).set_index('Institution')
        # dfMOCA = pd.DataFrame([['MOCA (severity)', 20, 2, 3.2, 4.8],
        df_moca = pd.DataFrame([['MOCA (severity)', 19, 2, 4.2, 4.8],
                               ['MOCA (cut-off)', 0, 0, 26, 4]], columns=[' ',
                                                                          'AD', 'AD ∩ MCI', 'MCI',
                                                                          'Norm. Contr.']).set_index(' ')
        plt.figtext(0.5,0.5, 'Class Label Differention by institution', ha="center", va="top", fontsize=16)
        plt.subplots_adjust(hspace = 0.5)
        axs[1,0].set_title('MMSE Differentiation')
        axs[1,1].set_title('MOCA Differentiation')
        df_mmse.plot.barh(stacked=True, ax=axs[1, 0]);
        df_moca.plot.barh(stacked=True, ax=axs[1, 1]);
