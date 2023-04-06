from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from sklearn.base import ClassifierMixin
from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np
from sklearn.base import clone
from pymoo.core.problem import ElementwiseProblem
from sklearn.model_selection import cross_val_score
import autograd.numpy as anp
from . import evaluate_model
from . import roc_plot


class FeatureSelectionAccuracyCostMultiProblem(ElementwiseProblem):
    def __init__(self, X, y, test_size, estimator, feature_names, cv, feature_costs, scale_features=0.5, objectives=2, random_state=0):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.estimator = estimator
        self.objectives = objectives
        self.L = feature_names
        self.n_max = len(self.L)
        self.scale_features = scale_features
        self.feature_costs = feature_costs
        self.cv = cv

        super().__init__(n_var=self.n_max, n_obj=objectives,
                         n_constr=1, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):

        clf = clone(self.estimator)
        selected = x > 0.5

        f1 = len(np.argwhere(selected == True))
        #f2=-1*np.mean(cross_val_score(clf, self.X[:,selected], self.y,cv=kf, scoring="accuracy"))
        num_features = self.n_max
        alpha = 0.99
        score = 1-np.mean(cross_val_score(estimator=clf,
                          X=self.X[:, selected], y=self.y, cv=self.cv, scoring="accuracy"))
        num_selected = len(np.argwhere(selected == True))
        f2 = alpha * score + (1 - alpha) * (num_selected / num_features)
        b = anp.column_stack(np.array([f1, f2]))
        out["F"] = b

        # Function constraint to select specific numbers of features:
        number = int((1 - self.scale_features) * self.n_max)
        out["G"] = (self.n_max - np.sum(x) - number)


class NSGAAccCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, scale_features=0.5, test_size=0.5, pareto_decision='accuracy', criteria_weights=None, objectives=2, p_size=100, c_prob=0.1, m_prob=0.1):
        self.base_estimator = base_estimator
        self.test_size = test_size
        self.p_size = p_size
        self.c_prob = c_prob
        self.m_prob = m_prob

        self.feature_costs = None
        self.estimator = None
        self.res = None
        self.selected_features = None
        self.fig_filename = None
        self.solutions = None
        self.pareto_decision = pareto_decision
        self.objectives = objectives
        self.scale_features = scale_features
        self.criteria_weights = criteria_weights

    def fit(self, X, y):
        features = range(X.shape[1])
        problem = FeatureSelectionAccuracyCostMultiProblem(
            X, y, self.test_size, self.base_estimator, features, self.feature_costs, self.scale_features, self.objectives)

        algorithm = NSGA2(
            pop_size=self.p_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True)

        res = minimize(
            problem,
            algorithm,
            ('n_eval', 1000),
            seed=1,
            verbose=False,
            save_history=True)
        print(res.F)
        print(res.X)
        # Select solution from the Pareto front
        # F returns all solutions in form [-accuracy, total_cost]
        self.solutions = res.F
        self.cols = res.X

        # X returns True and False which features has been selected
        if self.pareto_decision == 'accuracy':
            index = np.argmin(self.solutions[:, 1], axis=0)
            self.selected_features = res.X[index]
        elif self.pareto_decision == 'cost':
            index = np.argmin(self.solutions[:, 1], axis=0)
            self.selected_features = res.X[index]

        self.estimator = self.base_estimator.fit(
            X[:, self.selected_features], y)
        return self

    def predict(self, X):
        return self.estimator.predict(X[:, self.selected_features])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[:, self.selected_features])

    def selected_features_cost(self):
        total_cost = 0
        for id, cost in enumerate(self.feature_costs):
            if self.selected_features[id]:
                total_cost += cost
        return total_cost


def fit_nsga(X, y, clf, kf, output_path, p_size=14, scoring="accuracy"):
    X_ns = np.array(X)
    y = np.array(y)
    method = NSGAAccCost(clf, p_size=p_size)
    new_features = method.fit(X_ns, y)
    columns = X.columns.to_list()
    print(new_features.selected_features)
    res=evaluate_model.evaluate_classifier(X[np.array(columns)[new_features.selected_features]],
                                       y, classifier_model=clf, kfold_cv=kf, scoring_metric=scoring,output_path=output_path)

    roc_plot.plot_roc_kf(X=X[np.array(columns)[
                         new_features.selected_features]], y=y, classifier=clf)

    return res
