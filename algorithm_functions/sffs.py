from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from . import evaluate_model
from . import roc_plot
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sfs_fit_and_evaluate(model,X,y,cv, scoring, output_path,n_features=10):
    
    kf=cv
    sfs=SFS(model, k_features=n_features, forward=True, 
            floating=True, scoring=scoring, cv=kf)

    input_var=X.columns.to_list()


    sfs.fit(X,y)

    results=pd.DataFrame.from_dict(sfs.get_metric_dict()).T

    best_model=list(results["avg_score"]).index(max(results["avg_score"]))+1
    best_vars_n=list(results.iloc[best_model]["feature_names"])


    print(best_vars_n)

    res = evaluate_model.evaluate_classifier(X[best_vars_n],
                                                y, classifier_model=model, 
                                                kfold_cv=kf, scoring_metric=scoring, 
                                                output_path=output_path)
    return res
