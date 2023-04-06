import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization
import evaluate_model
import roc_plot


class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, kf, classifier, alpha=0.99,):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier
        self.alpha = alpha
        self.kf = kf

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(
            self.classifier, self.X_train[:, selected], self.y_train, self.kf).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)


def Particle(x, y, kf, classifier, scoring, output_path, p_size=20):
    X = np.array(x)
    y = np.array(y)
    feature_names = np.array(x.columns.to_list())

    problem = SVMFeatureSelection(X, y, classifier)
    task = Task(problem, max_iters=100)
    algorithm = ParticleSwarmOptimization(
        population_size=p_size, seed=42, n_jobs=-1, VERBOSE=True)
    best_features, best_fitness = algorithm.run(task)

    selected_features = best_features > 0.5
    print('Number of selected features:', selected_features.sum())
    print('Selected features:', ', '.join(
        feature_names[selected_features].tolist()))
    new_feat = feature_names[selected_features].tolist()

    x = X[:, selected_features]
    res = evaluate_model.evaluate_classifier(x,
                                             y, classifier_model=classifier, kfold_cv=kf, scoring_metric=scoring, output_path=output_path)

    roc_plot.plot_roc_kf(X=x, y=y, classifier=classifier)

    return res
