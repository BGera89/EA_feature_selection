import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict, GridSearchCV
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import KFold
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from . import evaluate_model
from . import roc_plot

def fit_and_evaluate(x, y, classifier, kfold, scoring, n_vars ,npop, output_path ):
        #Implementing the genetic algortihm
        model = GeneticSelectionCV(
                classifier, cv=kfold, verbose=1,
                scoring=scoring, max_features=n_vars,
                n_population=npop, crossover_proba=0.5,
                mutation_proba=0.2, n_generations=50,
                tournament_size=3, n_gen_no_change=10,
                caching=True,)
        model = model.fit(x, y)
        print('Features:', x.columns[model.support_])
        new_features=x.columns[model.support_]

        #training a classifier with the selected features
        #CV_rfc = GridSearchCV(estimator=classifier(random_state=42), param_grid=param_grid, cv= kf)
        #CV_rfc.fit(x[new_features],y)
        #ezt javítsd meg


        #cheching the scores with the new features
        scores_m=cross_val_predict(classifier, x[new_features], 
                y, cv=kfold)
        matrix=confusion_matrix(y, scores_m)
        print("Confusion matrix: ")
        matrix_labels=pd.DataFrame(matrix)
        matrix_labels=matrix_labels.rename(columns={0:"predicted 0", 1:"predicted 1"}, index={0:"actual 0", 1:"actual 1"})
        print(matrix_labels)
        print("\n")
        print("Model: ", classifier)
        
        res=evaluate_model.evaluate_classifier(x[new_features],
                                       y, classifier_model=classifier, kfold_cv=kfold, scoring_metric=scoring,output_path=output_path)

        return res




