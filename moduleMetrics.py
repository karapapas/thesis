import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve, confusion_matrix, plot_precision_recall_curve, make_scorer
from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack


def custom_measure(y_actual, y_hat):
    cm = confusion_matrix(y_actual, y_hat)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    return tp, fp, tn, fn


def custom_specificity(y_true, y_pred):
    tp, fp, tn, fn = custom_measure(y_true, y_pred)
    if tn + fp == 0:
        return 1
    else:
        return tn / (tn + fp)


class MetricsMethods:

    # Based on the example at https://scikit-learn.org/stable/modules/model_evaluation.html
    @staticmethod
    def generate_metrics(models, x_test, y_test, metrics, cv_num, show_raw_data):

        print("Count of label NC (id 2 in Database, 1 in CM) in y_test: {}".format(sum(y_test == 2)))
        print("Count of label AD-MCI (id 1 in Database, 0 in CM) in y_test: {} \n".format(sum(y_test == 1)))

        # for roc curves
        roc_axes = None

        # for precision recall curves
        pr_axes = None

        # for confusion matrix
        cf_matrix = dict.fromkeys(models.keys())

        x_test_m = x_test
        x_test_cm = x_test
        y_test_m = y_test
        y_test_cm = y_test
        for idx, (model_name, model) in enumerate(models.items()):
            if show_raw_data:
                print("\n", model_name + " Scores")

            # print given metrics in boxplot
            # boxplot example
            # https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
            mbox = pd.DataFrame()
            for metric in metrics:
                metric_name = metric
                if metric == 'specificity':
                    metric = make_scorer(custom_specificity)
                results = cross_val_score(model, x_test_m, y_test_m, cv=cv_num, scoring=metric)
                # print('results', type(results))
                if show_raw_data:
                    raw = np.array2string(results, threshold=np.inf, max_line_width=np.inf, separator=',')
                    raw.replace('\n', '').replace(' ', '')
                    print(metric, ": %.3f%% Std.: %.3f%%" % (results.mean() * 100.0, results.std() * 100.0))
                    # print("Metric raw data: ", raw)
                if isinstance(type(metric), str):
                    mbox[metric] = results
                else:
                    mbox[metric_name] = results

            # plot confusion matrix
            # based on the example at https://stackoverflow.com/questions/61825227/
            y_pred = model.predict(x_test_cm)
            cf_matrix[model_name] = confusion_matrix(y_test_cm, y_pred)

            fig, (axm, axc) = plt.subplots(1, 2, figsize=(18, 6))
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

    @staticmethod
    def print_decision_surface(models, data_set_tuples, show_probabilities):

        for idx, (model_name, model) in enumerate(models.items()):
            fig, axs = plt.subplots(1, 3, figsize=(18, 4))
            fig.suptitle(model_name, fontsize=14)

            column_counter = 0
            column_titles = ['Train Set', 'Test Set', 'Entire Set']
            for ds in data_set_tuples:
                axs[column_counter].set_title(column_titles[column_counter])

                x = ds[0]
                y = ds[1]

                # define bounds of the domain
                min1, max1 = x[:, 0].min() - 1, x[:, 0].max() + 1
                min2, max2 = x[:, 1].min() - 1, x[:, 1].max() + 1

                # define the x and y scale
                x1grid = arange(min1, max1, 0.1)
                x2grid = arange(min2, max2, 0.1)

                # create all of the lines and rows of the grid
                xx, yy = meshgrid(x1grid, x2grid)

                # flatten each grid to a vector
                r1, r2 = xx.flatten(), yy.flatten()
                r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

                # horizontal stack vectors to create x1,x2 input for the model
                grid = hstack((r1, r2))

                if show_probabilities is True:
                    y_hat = model.predict_proba(grid)
                    # keep just the probabilities for class 0
                    y_hat = y_hat[:, 0]
                else:
                    y_hat = model.predict(grid)

                # reshape the predictions back into a grid
                zz = y_hat.reshape(xx.shape)

                # plot the grid of x, y and z values as a surface
                # c = plt.contourf(xx, yy, zz, c_map='binary')
                axs[column_counter].contourf(xx, yy, zz, cmap='binary')

                # add a legend, called a color bar
                # plt.color_bar(c)

                # create scatter plot for samples from each class
                for class_value in range(1, 3):
                    # get row indexes for samples with this class
                    # row_ix = where(smote_y == class_value)
                    row_ix = where(y == class_value)

                    # create scatter of these samples
                    # plt.scatter(x[row_ix, 0], x[row_ix, 1], c_map='Paired')
                    axs[column_counter].scatter(x[row_ix, 0], x[row_ix, 1], cmap='Paired')

                # show the plot
                # plt.show()
                column_counter += 1
