from sklearn.model_selection import cross_val_predict, GridSearchCV
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import KFold
from sklearn.metrics import RocCurveDisplay
import evaluate_model


def new_estimator(x, y, classifier, kfold, param_grid, njobs=1):

    CV = GridSearchCV(x, y, estimator=classifier,
                      param_grid=param_grid, cv=kfold, n_jobs=njobs)
    CV.fit(x, y)
    print("with features: ", x.columns.to_list)
    print(CV.best_estimator_)
    print(CV.best_score_)
    print(CV.best_params_)
    evaluate_model.evaluate_new(
        x, y, scoring="accuracy", classifier=CV.best_estimator_, )
    return CV.best_estimator_
