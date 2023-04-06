import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict, GridSearchCV
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import KFold
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from . import roc_plot


def evaluate_classifier(x, y, scoring_metric, classifier_model, kfold_cv, output_path):
    """
    Evaluate the performance of a classifier model using cross-validation.

    Parameters:
        x (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        scoring_metric (str): Scoring metric to use for cross-validation.
        classifier_model: Classifier model to evaluate.
        kfold_cv: Cross-validation strategy.

    Returns:
        Tuple containing the list of column names for `x`, the mean score, sensitivity, specificity, ROC AUC score, and the `classifier_model` object.
    """
    # Generate cross-validated predictions
    predicted_y = cross_val_predict(classifier_model, x, y, cv=kfold_cv)

    # Compute confusion matrix and print it
    conf_matrix = confusion_matrix(y, predicted_y)
    conf_matrix_labels = pd.DataFrame(conf_matrix, columns=[
                                      "Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
    print("Confusion matrix: ")
    print(conf_matrix_labels)
    print("\n")

    # Compute cross-validated scores and ROC AUC score
    scores = cross_val_score(classifier_model, x, y,
                             cv=kfold_cv, scoring=scoring_metric)
    roc_auc = cross_val_score(classifier_model, x, y,
                              cv=kfold_cv, scoring="roc_auc")

    # Compute sensitivity and specificity
    sensitivity = conf_matrix_labels.loc["Actual 1",
                                         "Predicted 1"] / conf_matrix_labels.loc["Actual 1"].sum()
    specificity = conf_matrix_labels.loc["Actual 0",
                                         "Predicted 0"] / conf_matrix_labels.loc["Actual 0"].sum()

    # Print results and plot ROC curve
    print("Mean score is: ", scores.mean())
    print("Sensitivity is: ", sensitivity)
    print("Specificity is: ", specificity)
    print("ROC AUC score is: ", roc_auc.mean())
    print(scores)
    roc_plot.plot_roc_kf(X=x, y=y, classifier=classifier_model, cv=kfold_cv)

    results_dict = {
        "Features": x.columns.to_list(),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": scores.mean(),
        "ROC AUC": roc_auc.mean(),
        "Classifier": str(classifier_model)
    }

    df_results = pd.DataFrame.from_dict(results_dict)
    print('choosen features: ', x.columns.to_list())

    # Saving the results to a csv file
    df_results.to_csv(output_path, index=False)

    # Return results and classifier object
    return x.columns.to_list(), scores.mean(), sensitivity, specificity, roc_auc.mean(), classifier_model
