import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix

class MetricsMethods:

    # Based on the example at https://scikit-learn.org/stable/modules/model_evaluation.html
    def generateMetrics(self, models, X, y, metrics):
        axes = None
        for idx, model in enumerate(models):
            for metric in metrics:
                results = cross_val_score(model, X, y, cv=5, scoring=metric)
                raw = np.array2string(results, threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', '')
                # print("Metric: ", metric, " Score: %.3f%% Std.: %.3f%% Raw Results: %s" % (results.mean() * 100.0, results.std() * 100.0, raw))
                print("Metric: ", metric, " Score: %.3f%% Std.: %.3f%%" % (results.mean() * 100.0, results.std() * 100.0))
            if idx == 0:
                display = plot_roc_curve(model, X, y)
                axes = display.ax_
            else:
                plot_roc_curve(model, X, y, ax=axes)

    # TODO
    #  include prediction probabilities
    #  here for the exploratory analysis
    #  and in service for the user
    #  clf.predict_proba(X_test)