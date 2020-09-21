from sklearn.model_selection import cross_val_score

class MetricsMethods:

    '''
    Based on the example at
    https://scikit-learn.org/stable/modules/model_evaluation.html
    '''
    def generateMetrics(model, X, y, metrics):
        print('in generate metrics', metrics)
        for scoringMetric in metrics:
            print(cross_val_score(model, X, y, cv=5, scoring=scoringMetric))
