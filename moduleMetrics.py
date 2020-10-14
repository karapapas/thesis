from sklearn.model_selection import cross_val_score
import numpy as np

class MetricsMethods:

    # plot_roc_curve(model_LR, X_test, y_test)
    # # disp = plot_roc_curve(model_LR, X_test, y_test)
    # # plot_roc_curve(model_DTC, X_test, y_test, ax=disp.ax_)
    # # plot_roc_curve(model_RFC, X_test, y_test, ax=disp.ax_)

    # Based on the example at https://scikit-learn.org/stable/modules/model_evaluation.html
    def generateMetrics(self, model, X, y, metrics):

        for metric in metrics:
            results = cross_val_score(model, X, y, cv=5, scoring=metric)
            raw = np.array2string(results, threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', '')
            print("Metric: ", metric, " Score: %.3f%% Std.: %.3f%% Raw Results: %s" % (results.mean() * 100.0,
                                                                                       results.std() * 100.0,
                                                                                       raw))

    # TODO
    #  include prediction probabilities
    #  here for the exploratory analysis
    #  and in service for the user
    #  clf.predict_proba(X_test)