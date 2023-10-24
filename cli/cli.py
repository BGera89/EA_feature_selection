import pandas as pd
import numpy as np
import argparse
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from algorithm_functions import simulated_annaeling
from algorithm_functions import pso
from algorithm_functions import nsga_II
from algorithm_functions import genetic_alg
from algorithm_functions import sffs
from algorithm_functions import evaluate_model
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from algorithm_functions import train_new_estimator




# Setting up the arguments for the CLI
parser = argparse.ArgumentParser(
    description="Python app for feature selection with evolutionary algorithms")

# General arguments
parser.add_argument('-fp', '--filepath', type=str,
                    help='Type in the filepath of the .csv file with your dataset. Default is \'dataset_dir\X.csv\' ',
                    default='dataset_dir/X.csv')
parser.add_argument('-mt', '--model_type', type=str,
                    help='Type of the model you want to use. Default = SVC ',
                    default='SVC', choices=['SVC', 'RF',])
parser.add_argument('-fs', '--feature_selector', type=str,
                    help='Type of feature selection algorithm you want to use',
                    default='GA', choices=['GA', 'NSGA', 'PSO', 'SA', 'SFFS'])
parser.add_argument('-l', '--label', type=str,
                    help='name of the column that contrains the label (the y, or the value we are predicting). Default is quality',
                    default='quality')
parser.add_argument('-s', '--split', type=int,
                    help='The number of the k-fold splits. Default is: 5',
                    default=5)
parser.add_argument('-tf', '--train_features', type=str, nargs='*',
                    help='Features (columns of csv) for the model to make the predictions. Default uses all. Write your columns names afer each other. e.g. -mf pH sulphates. If there the feature consists of multiple words write it in double quotes like this: \"fixed acidity\". If the number of feature is one or two it will create visualizations for those (2D if only one feature is present, 3D if two features are present)')
parser.add_argument('-sc', '--scoring', type=str,
                    help='The type of critera you want to train your model on. Default = accuracy',
                    default='accuracy')
parser.add_argument('-op', '--output_path', type=str,
                    help='Specify the path of the output file. It will contain the evaluation metrics, and the predicted values with true values in a separate file. Default is the currect dir',
                    default='output')
parser.add_argument('-nn', '--not_needed', type=int,
                    help='Cuts the not needed columns from the DataFrame (like the first ID column). Default = 27',
                    default=1)
parser.add_argument('-nne', '--not_needed_end', type=int,
                    help='Cuts the not needed columns from the DataFrame (like the first ID column). Default = 27',
                    default=-1)
parser.add_argument('-ho', '--hyperparameter_opt', type=bool,
                    help='Cuts the not needed columns from the DataFrame (like the first ID column). Default = 27',
                    default=False)

# SVM classifier's arguments
parser.add_argument('-sk', '--svm_kernel', type=str,
                    help='The typeof the kernel for the SVR model. Default is rbf', choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    default='rbf')
parser.add_argument('-de', '--degree', type=int,
                    help='The degree of the polynomial if the kernel is \'poly\' (ignored by other kernels). Default is 3.',
                    default=3)
parser.add_argument('-ga', '--gamma', type=str,
                    help='Kernel coefficient for \'rbf\', \'poly\' and \'sigmoid\'. Default is scale',
                    choices=['scale', 'auto'],
                    default='scale')
parser.add_argument('-co', '--coef0', type=float,
                    help='Independent term in kernel function. It is only significant in \'poly\' and \'sigmoid\'. Default is 0.0',
                    default=0.0)
parser.add_argument('-to', '--tol', type=float,
                    help='Tolerance for stopping criterion. Default is 1e-3',
                    default=1e-3)
parser.add_argument('-c', '--C', type=float,
                    help='Regularization parameter The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty. Default is 1.0',
                    default=1.0)
parser.add_argument('-e', '--epsilon', type=float,
                    help='Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value. Must be non-negative Default is 0.1.',
                    default=0.1)
parser.add_argument('-sh', '--shrinking', type=bool,
                    help='Whether to use the shrinking heuristic. See the User Guide. Default is True',
                    default=True)
parser.add_argument('-v', '--verbose', type=bool,
                    help='Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context. Default is False',
                    default=False)

# RF algorithms arguments
parser.add_argument('--n_estimators', type=int, default=100,
                    help='The number of trees in the forest')
parser.add_argument('--criterion', type=str, default='gini',
                    choices=['gini', 'entropy'],
                    help='The function to measure the quality of a split')
parser.add_argument('--max_depth', type=int, default=None,
                    help='The maximum depth of the tree')
parser.add_argument('--min_samples_split', type=int, default=2,
                    help='The minimum number of samples required to split an internal node')
parser.add_argument('--min_samples_leaf', type=int, default=1,
                    help='The minimum number of samples required to be at a leaf node')
parser.add_argument('--max_features', type=str, default='sqrt',
                    choices=['sqrt', 'log2', None],
                    help='The number of features to consider when looking for the best split')
parser.add_argument('--bootstrap', type=bool, default=True,
                    help='Whether to bootstrap samples when building trees')
parser.add_argument('--class_weight', type=str, default=None,
                    help='Weights associated with classes in case of class imbalance')


# FS algorithm arguments
parser.add_argument('-nv', '--n_vars', type=int,
                    help='number of variables to choose from. Default=10',
                    default=31)
parser.add_argument('-np', '--number_population', type=int,
                    help='The number of the population for each algorithm. Default=100',
                    default=500)

#Iterations
parser.add_argument('-ib', '--iteration_true', type=bool,
                    help='decide if you want to run the algorithms iteratively',
                    default=False)

parser.add_argument('-it', '--iterations', type=int,
                    help='The number of iterations for each algorithm. Default=10',
                    default=10)


args = parser.parse_args()


df = pd.read_csv(args.filepath)
X = df.drop(columns=[args.label])
X = X.iloc[:, args.not_needed:args.not_needed_end]
y = df[args.label]

# cross validation with a split
kf = KFold(n_splits=args.split, shuffle=True, random_state=42)


if args.model_type == 'SVC':
    model = SVC(kernel=args.svm_kernel, degree=args.degree, gamma=args.gamma, coef0=args.coef0,
                tol=args.tol, C=args.C, shrinking=args.shrinking, verbose=args.verbose)

    param_grid={
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1],
    'degree': [2, 3, 4],
    'coef0': [0.0, 1.0, 2.0]
    }
else:
    model = RandomForestClassifier(n_estimators=args.n_estimators, criterion=args.criterion,
                                    max_depth=args.max_depth, min_samples_split=args.min_samples_split,
                                    min_samples_leaf=args.min_samples_leaf, max_features=args.max_features,
                                    bootstrap=args.bootstrap, class_weight=args.class_weight, random_state=42,
                                    n_jobs=-1)
    param_grid= {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
    }
if args.iteration_true==True:
    for i in range(args.iterations):
        if args.feature_selector == 'GA':
            genetic_alg.fit_and_evaluate(
                X, y, model, kfold=kf, scoring=args.scoring, 
                n_vars=args.n_vars, npop=args.number_population, 
                output_path=args.output_path+'_GA_iter'+str(i)+'_'+args.model_type+'.csv')

        if args.feature_selector == 'NSGA':
            nsga_II.fit_nsga(
                X=X, y=y, clf=model, kf=kf, 
                output_path=args.output_path +'_NSGA_iter'+str(i)+'_'+args.model_type+'.csv', 
                p_size=args.number_population, scoring=args.scoring)

        if args.feature_selector == 'PSO':
            pso.Particle(x=X, y=y, kf=kf, classifier=model, 
                        scoring=args.scoring,
                        output_path=args.output_path+'_PSO_iter'+str(i)+'_'+args.model_type+'.csv', 
                        p_size=args.number_population)
        if args.feature_selector == 'SA':
            simulated_annaeling.simulated_annealing(
                X, y, model, kf, args.scoring, args.output_path+'_SA_iter'+str(i)+'_'+args.model_type+'.csv')
            
        if args.feature_selector == 'SFFS':
            sffs.sfs_fit_and_evaluate(
                X=X, y=y, model=model,cv=kf, scoring=args.scoring,
                output_path=args.output_path+'_SFFS_iter'+str(i)+'_'+args.model_type+'.csv')

else:
    if args.feature_selector == 'GA':
        genetic_alg.fit_and_evaluate(
            X, y, model, kfold=kf, scoring=args.scoring, 
            n_vars=args.n_vars, npop=args.number_population, 
            output_path=args.output_path+'_'+args.model_type+'_GA.csv')

    if args.feature_selector == 'NSGA':
        nsga_II.fit_nsga(
            X=X, y=y, clf=model, kf=kf, 
            output_path=args.output_path +'_'+args.model_type+'_NSGA.csv', 
            p_size=args.number_population, scoring=args.scoring)

    if args.feature_selector == 'PSO':
        pso.Particle(x=X, y=y, kf=kf, classifier=model, 
                    scoring=args.scoring,
                    output_path=args.output_path+'_'+args.model_type+'_PSO.csv', 
                    p_size=args.number_population)
    if args.feature_selector == 'SA':
        simulated_annaeling.simulated_annealing(
            X, y, model, kf, args.scoring, args.output_path+'_'+args.model_type+'_SA.csv')
        
    if args.feature_selector == 'SFFS':
        sffs.sfs_fit_and_evaluate(
            X=X, y=y, model=model,cv=kf, scoring=args.scoring,
            output_path=args.output_path+'_'+args.model_type+'_SFFS.csv')

    if args.hyperparameter_opt == True:
        features=pd.read_csv(args.output_path+'_'+args.model_type+'_'+args.feature_selector+'.csv')
        train_new_estimator.new_estimator(x=X[features['Features']], 
                                          y=y,
                                          classifier=model,
                                          kfold=kf,
                                          param_grid_est=param_grid,
                                          output_path_=args.output_path+'_'+args.model_type+'_'+args.feature_selector+'param_opt.csv')
