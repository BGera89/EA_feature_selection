import random
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import sys
import os
from . import evaluate_model
from . import roc_plot


def train_model(X, y, classifier, cv, scoring="accuracy"):
    """
    Run random forest classification model on feature subset
    and retrieve cross validated ROC-AUC score
    """
    clf = classifier
    kfold = cv
    cv_score = cross_val_score(clf, X, y, cv=kfold,
                               scoring=scoring,)

    return cv_score.mean()


def simulated_annealing(X_train,
                        y_train, classifier,
                        cv, scoring,
                        output_path,
                        maxiters=500,
                        alpha=0.99,
                        beta=1,
                        T_0=10,
                        update_iters=1,
                        subset_percent=0.5,
                        temp_reduction='geometric'):
    columns = ['Iteration', 'Feature Count', 'Feature Set', 'Metric', 'Best Metric',
               'Acceptance Probability', 'Random Number', 'Outcome']
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_subset = None
    hash_values = set()
    T = T_0
    full_set = set(np.arange(len(X_train.columns)))

    # Generate initial random subset based on ~50% of columns
    curr_subset = set(random.sample(
        list(full_set), round(subset_percent * len(full_set))))

    print(curr_subset)
    X_curr = X_train.iloc[:, list(curr_subset)]
    prev_metric = train_model(X_curr, y_train, classifier, cv=cv)
    best_metric = prev_metric

    for i in range(maxiters):
        if T < 0.01:
            print(
                f'Temperature {T} below threshold. Termination condition met')
            break

        while True:
            if len(curr_subset) == len(full_set):
                move = 'Remove'
            elif len(curr_subset) == 2:  # Not to go below 2 features
                move = random.choice(['Add', 'Replace'])
            else:
                move = random.choice(['Add', 'Replace', 'Remove'])

            pending_cols = full_set.difference(curr_subset)
            new_subset = curr_subset.copy()

            if move == 'Add':
                new_subset.add(random.choice(list(pending_cols)))
            elif move == 'Replace':
                new_subset.remove(random.choice(list(curr_subset)))
                new_subset.add(random.choice(list(pending_cols)))
            else:
                new_subset.remove(random.choice(list(curr_subset)))

            if new_subset in hash_values:
                print('Subset already visited')
            else:
                hash_values.add(frozenset(new_subset))
                break

        X_new = X_train.iloc[:, list(new_subset)]
        metric = train_model(X_new, y_train, classifier, cv=cv)

        if metric > prev_metric:
            print('Local improvement in metric from {:8.4f} to {:8.4f} '
                  .format(prev_metric, metric) + ' - New subset accepted')
            outcome = 'Improved'
            accept_prob, rnd = '-', '-'
            prev_metric = metric
            curr_subset = new_subset.copy()

            if metric > best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} '
                      .format(best_metric, metric) + ' - Best subset updated')
                best_metric = metric
                best_subset = new_subset.copy()

        else:
            rnd = np.random.uniform()
            diff = prev_metric - metric
            accept_prob = np.exp(-beta * diff / T)

            if rnd < accept_prob:
                print('New subset has worse performance but still accept. Metric change' +
                      ':{:8.4f}, Acceptance probability:{:6.4f}, Random number:{:6.4f}'
                      .format(diff, accept_prob, rnd))
                outcome = 'Accept'
                prev_metric = metric
                curr_subset = new_subset.copy()
            else:
                print('New subset has worse performance, therefore reject. Metric change' +
                      ':{:8.4f}, Acceptance probability:{:6.4f}, Random number:{:6.4f}'
                      .format(diff, accept_prob, rnd))
                outcome = 'Reject'

        results.loc[i, 'Iteration'] = i+1
        results.loc[i, 'Feature Count'] = len(curr_subset)
        results.loc[i, 'Feature Set'] = sorted(curr_subset)
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric
        results.loc[i, 'Acceptance Probability'] = accept_prob
        results.loc[i, 'Random Number'] = rnd
        results.loc[i, 'Outcome'] = outcome

        # Temperature cooling schedule
        if i % update_iters == 0:
            if temp_reduction == 'geometric':
                T = alpha * T
            elif temp_reduction == 'linear':
                T -= alpha
            elif temp_reduction == 'slow decrease':
                b = 5  # Arbitrary constant
                T = T / (1 + b * T)
            else:
                raise Exception(
                    "Temperature reduction strategy not recognized")

    best_subset_cols = [list(X_train.columns)[i] for i in list(best_subset)]
    results = results.dropna(axis=0, how='all')

    print("results", results)
    x = X_train[best_subset_cols]
    y = y_train

    res = evaluate_model.evaluate_classifier(x,
                                             y, classifier_model=classifier, kfold_cv=cv, scoring_metric=scoring, output_path=output_path)

    return res
