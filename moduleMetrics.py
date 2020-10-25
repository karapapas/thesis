import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, confusion_matrix, plot_precision_recall_curve


class MetricsMethods:

    # Based on the example at https://scikit-learn.org/stable/modules/model_evaluation.html
    @staticmethod
    def generate_metrics(models, x_test, y_test, metrics, cv_num, show_raw_data):

        # for roc curves
        roc_axes = None

        # for precision recall curves
        pr_axes = None

        # for confusion matrix
        cf_matrix = dict.fromkeys(models.keys())

        # for idx, model in enumerate(models):
        for idx, (model_name, model) in enumerate(models.items()):

            # print given metrics in boxplot
            # boxplot example
            # https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
            mbox = pd.DataFrame()
            for metric in metrics:
                results = cross_val_score(model, x_test, y_test, cv=cv_num, scoring=metric)
                if show_raw_data:
                    raw = np.array2string(results, threshold=np.inf, max_line_width=np.inf, separator=',')
                    raw.replace('\n', '').replace(' ', '')
                    print(metric, " Score: %.3f%% Std.: %.3f%%" % (results.mean() * 100.0, results.std() * 100.0))
                    print("Metric raw data: ", raw)
                mbox[metric] = results

            # plot confusion matrix
            # based on the example at https://stackoverflow.com/questions/61825227/
            y_pred = model.predict(x_test)
            cf_matrix[model_name] = confusion_matrix(y_test, y_pred)

            # specificity
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            mbox['Specificity'] = (tn / (tn + fp))

            fig, (axm, axc) = plt.subplots(1, 2, figsize=(12, 2))
            boxplot_title = 'Metrics for ' + model_name
            sns.boxplot(data=mbox, orient="h", palette="Set3", showmeans=True, ax=axm).set_title(boxplot_title)
            sns.heatmap(cf_matrix[model_name], ax=axc, annot=True, fmt='d', cmap="RdBu")

        for idx, (model_name, model) in enumerate(models.items()):
            # plot roc curves
            if idx == 0:
                display_roc = plot_roc_curve(model, x_test, y_test)
                # print(type(display_roc))
                roc_axes = display_roc.ax_
            else:
                plot_roc_curve(model, x_test, y_test, ax=roc_axes)


            # plot precision recall curves
            if idx == 0:
                display_pr = plot_precision_recall_curve(model, x_test, y_test)
                pr_axes = display_pr.ax_
            else:
                plot_precision_recall_curve(model, x_test, y_test, ax=pr_axes)

        roc_axes.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        pr_axes.plot([0, 1], [0.5, 0.5], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)