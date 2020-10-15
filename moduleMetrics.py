import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, confusion_matrix


class MetricsMethods:

    # Based on the example at https://scikit-learn.org/stable/modules/model_evaluation.html
    @staticmethod
    def generate_metrics(models, x_test, y_test, metrics, show_raw_data, ):

        roc_axes = None     # for roc curves
        cf_matrix = dict.fromkeys(models.keys())    # for confusion matrix

        # for idx, model in enumerate(models):
        for idx, (model_name, model) in enumerate(models.items()):

            # print given metrics
            print("Metrics for ", model_name)
            for metric in metrics:
                results = cross_val_score(model, x_test, y_test, cv=3, scoring=metric)
                raw = np.array2string(results, threshold=np.inf, max_line_width=np.inf, separator=',')
                raw.replace('\n', '').replace(' ', '')
                print(metric, " Score: %.3f%% Std.: %.3f%%" % (results.mean() * 100.0, results.std() * 100.0))
                if show_raw_data:
                    print("Metric raw data: ", raw)
            print("############## \n")

            # plot roc curves
            if idx == 0:
                display = plot_roc_curve(model, x_test, y_test)
                roc_axes = display.ax_
            else:
                plot_roc_curve(model, x_test, y_test, ax=roc_axes)

            # plot confusion matrix
            # based on the example at https://stackoverflow.com/questions/61825227/
            y_pred = model.predict(x_test)
            cf_matrix[model_name] = confusion_matrix(y_test, y_pred)

        fig, axn = plt.subplots(1, len(models), sharex=True, sharey=True, figsize=(12, 2))
        for i, ax in enumerate(axn.flat):
            k = list(cf_matrix)[i]
            sns.heatmap(cf_matrix[k], ax=ax, cbar=i == len(models), annot=True, fmt="d", cmap="RdBu")
            ax.set_title(k, fontsize=8)

        # Precision-Recall

# TODO include method for probabilities prediction for the prediction service clf.predict_proba(X_test)
