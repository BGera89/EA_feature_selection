import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization


class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, cv, classifier, alpha=0.99,):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier
        self.alpha = alpha
        self.cv = cv

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(
            self.classifier, self.X_train[:, selected], self.y_train, self.cv,).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)


def Particle(x, y, cv, classifier, p_size=20):
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
    scores_m = cross_val_predict(classifier, x,
                                 y, cv)
    matrix = confusion_matrix(y, scores_m)
    print("Confusion matrix: ")
    matrix_labels = pd.DataFrame(matrix)
    matrix_labels = matrix_labels.rename(
        columns={0: "predicted 0", 1: "predicted 1"}, index={0: "actual 0", 1: "actual 1"})
    print(matrix_labels)
    print("\n")

    scores = cross_val_score(classifier, x,
                             y, cv, scoring="accuracy")
    roc_auc = cross_val_score(classifier, x,
                              y, cv, scoring="roc_auc")
    print("means score is: ", scores.mean())
    print("sensitivity is: ", matrix_labels["predicted 1"][1]/(
        matrix_labels["predicted 1"][1]+matrix_labels["predicted 0"][1]))
    print("specificity is: ", matrix_labels["predicted 0"][0]/(
        matrix_labels["predicted 0"][0]+matrix_labels["predicted 1"][0]))
    print("ROC AUC score is: ", roc_auc.mean())
    print(scores)
    return new_feat
