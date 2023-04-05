import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict, GridSearchCV
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import KFold
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix


def evaluate_new(x, y, scoring, classifier, kfold):

    scores_m = cross_val_predict(classifier, x,
                                 y, cv=kfold)
    matrix = confusion_matrix(y, scores_m)
    print("Confusion matrix: ")
    matrix_labels = pd.DataFrame(matrix)
    matrix_labels = matrix_labels.rename(
        columns={0: "predicted 0", 1: "predicted 1"}, index={0: "actual 0", 1: "actual 1"})
    print(matrix_labels)
    print("\n")

    scores = cross_val_score(classifier, x,
                             y, cv=kfold, scoring=scoring)
    roc_auc = cross_val_score(classifier, x,
                              y, cv=kfold, scoring="roc_auc")
    print("means score is: ", scores.mean())
    print("sensitivity is: ", matrix_labels["predicted 1"][1]/(
        matrix_labels["predicted 1"][1]+matrix_labels["predicted 0"][1]))
    print("specificity is: ", matrix_labels["predicted 0"][0]/(
        matrix_labels["predicted 0"][0]+matrix_labels["predicted 1"][0]))
    print("ROC AUC score is: ", roc_auc.mean())
    print(scores)
