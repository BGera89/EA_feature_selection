from sklearn.model_selection import cross_val_predict, GridSearchCV
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import KFold
from sklearn.metrics import RocCurveDisplay
from . import evaluate_model
from . import roc_plot


def new_estimator(x, y, classifier, kfold, 
                  param_grid_est, output_path_, njobs=8):
    print(classifier)
    CV = GridSearchCV(estimator=classifier,
                      param_grid=param_grid_est, cv=kfold, n_jobs=njobs)
    CV.fit(x, y)
    print("with features: ", x.columns.to_list)
    print(CV.best_estimator_)
    print(CV.best_score_)
    print(CV.best_params_)
    evaluate_model.evaluate_classifier(
        x, y, scoring_metric="accuracy", 
        classifier_model=CV.best_estimator_,
         kfold_cv=kfold, output_path=output_path_, )
    return CV.best_estimator_
