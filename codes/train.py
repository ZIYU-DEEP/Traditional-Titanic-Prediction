"""
Title:       Build a training pipeline that helps user fit models, tune
             hyperparameters, evaluate best models and save results.

Description:

Author:      Kunyu He, CAPP'20, The University of Chicago
"""

import shutil
import warnings
import argparse
import itertools
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, BaggingClassifier,
                              ExtraTreesClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from featureEngineering import ask, read_feature_names
from trainviz import (plot_predicted_scores, plot_precision_recall,
                      plot_auc_roc, plot_feature_importances)

# ----------------------------------------------------------------------------#
RAW_DATA_DIR = "../data/"
INPUT_DIR = "../processed_data/"
OUTPUT_DIR = "../evaluations/"
LOG_DIR = "../logs/train/logs/"
PREDICTED_PROBS_DIR = "../logs/train/predicted_probas/"
PREDICTIONS_DIR = "../logs/train/predictions/"
VIZ_DIR = "../logs/train/viz/"

# logging
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)
fh = logging.FileHandler(LOG_DIR + time.strftime("%Y%m%d-%H%M%S") + '.log')
logger.addHandler(fh)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------#
def load_features(labeled_test=False):
    """
    Load pre-processed feature matrices.

    Inputs:
        - labeled_test (bool): whether test data is labelled

    Returns:
        (N*m array) X_train, (n*m array) X_test,
        (N*1 array) y_train, (n*1 array or None) X_train
            where N is number of training observations, n is number of test
            observations. m is number of features.

    """
    X_train = np.load(INPUT_DIR + 'X_train.npy')
    y_train = np.load(INPUT_DIR + 'y_train.npy')
    X_test = np.load(INPUT_DIR + 'X_test.npy')

    if labeled_test:
        y_test = np.load(INPUT_DIR + 'y_test.npy')
    else:
        y_test = None

    return X_train, X_test, y_train, y_test


def create_dirs(dir_path):
    """
    Create a new directory if it doesn't exist and add a '.gitkeep' file to the
    directory.

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        file = open(dir_path + ".gitkeep", "w+")
        file.close()


def start_clean():
    """
    Wipe all the folders and documents (predicted probabilities, visualizations,
    predictions, and evaluations the modeling pipeline produces and start clean
    as requested.

    """
    to_clean = [OUTPUT_DIR, PREDICTIONS_DIR, PREDICTIONS_DIR, VIZ_DIR,
                PREDICTED_PROBS_DIR]
    for dir_path in to_clean:
        shutil.rmtree(dir_path)
        create_dirs(dir_path)


class Precision:
    """
    Constructed with given relative population threshold k to calculate
    precision at k.

    """

    def __init__(self, k):
        """
        Construct a class to calculate model precision at population threshold
        k.

        Inputs:
            - k (int): population threshold k, where k is the percentage of
                population at the highest probabilities to be classified as
                "positive".

        """
        self.k = k
        self.name = "Precision at {}%".format(k)

    def precision_at_k(self, y_val, predicted_prob):
        """
        Predict based on predicted probabilities and population threshold k,
        where k is the percentage of population at the highest probabilities to
        be classified as "positive". Label those with predicted probabilities
        higher than (1- k/100) quantile as positive, and evaluate the precision.

        Inputs:
            - predicted_prob (array): predicted probabilities on the validation
                set.
            - y_val (array): of true labels.

        Returns:
            (float) precision score of our model at population threshold k.

        """
        cut_off = np.quantile(predicted_prob, (1 - self.k / 100.0))
        labels = [1 if prob >= cut_off else 0 for prob in predicted_prob]

        return precision_score(y_val, labels)


class ModelingPipeline:
    """
    Modeling pipeline for build machine learning models on preprocessed training
    data as NumPy arrays, evaluate them with cross validation or on test set,
    and make predictions. Modify the class variables to add machine learning
    models, metrics, absolute decision thresholds and relative population
    thresholds, hyperparamter grids to search through.

    """

    MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest",
                   "Bagging", "Ada Boosting", "Gradient Boosting",
                   "Extra Tree", "Naive Bayes", "KNN", "Linear SVM"]
    MODELS = [LogisticRegression, DecisionTreeClassifier,
              RandomForestClassifier, BaggingClassifier, AdaBoostClassifier,
              GradientBoostingClassifier, ExtraTreesClassifier, GaussianNB,
              KNeighborsClassifier, LinearSVC]

    METRICS_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    METRICS = [accuracy_score, precision_score, recall_score, f1_score,
               roc_auc_score]

    DECISION_THRESHOLDS = [round(x, 1) for x in np.arange(0.1, 1, 0.1)]

    POPULATION_THRESHOLDS = [1, 2, 5, 10, 20, 30, 50]
    for k in POPULATION_THRESHOLDS:
        METRICS.append(Precision(k).precision_at_k)
        METRICS_NAMES.append(Precision(k).name)

    SEED = 123

    GRID_SEARCH_PARAMS = {
        "Logistic Regression": {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag']
        },

        "Decision Tree": {
            'min_samples_split': np.arange(0.01, 0.11, 0.01),
            'max_depth': [1, 5, 10, 15, 20]
        },

        "Random Forest": {
            'min_samples_split': np.arange(0.01, 0.11, 0.01),
            'max_depth': [1, 5, 10, 15, 20],
            'max_features': None
        },

        "Bagging": {
            'max_samples': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5],
            'max_features': None
        },

        "Ada Boosting": {
            'algorithm': {"SAMME", "SAMME.R"},
            'learning_rate': [0.001, 0.01, 0.1, 0.5, 1, 10]
        },

        "Gradient Boosting": {
            'learning_rate': [0.001, 0.01, 0.1, 0.5, 1, 10],
            'subsample': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5],
            'max_depth': [1, 5, 10, 15, 20]
        },

        "Extra Tree": {
            'min_samples_split': np.arange(0.01, 0.11, 0.01),
            'max_depth': [1, 5, 10, 15, 20],
            'max_features': None
        },

        "Naive Bayes": {},

        "KNN": {
            'n_neighbors': list(range(5, 55, 5)),
            'weights': ["uniform", "distance"],
            'metric': ["euclidean", "manhattan", "chebyshev", "minkowski"]
        },

        "Linear SVM": {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
        }
    }

    DEFAULT_ARGS = {"Logistic Regression": {'random_state': SEED},
                    "Decision Tree": {'random_state': SEED},
                    "Random Forest": {'n_estimators': 5000,
                                      'random_state': SEED,
                                      'oob_score': True, 'n_jobs': -1},
                    "Bagging": {'n_estimators': 1000, 'random_state': SEED,
                                'oob_score': True, 'n_jobs': -1},
                    "Ada Boosting": {'n_estimators': 1000, 'random_state': SEED},
                    "Gradient Boosting": {'n_estimators': 1000,
                                          'random_state': SEED},
                    "Extra Tree": {'n_estimators': 1000, 'random_state': SEED,
                                   'n_jobs': -1},
                    "Naive Bayes": {},
                    "KNN": {'n_jobs': -1},
                    "Linear SVM": {'random_state': SEED}
                    }

    def __init__(self, cv=5, ask_user=True, verbose=1, plot=True):
        """
        Construct a preprocessing pipeline given name of the data file.

        Inputs:
            - cv (int): number of folds for cross-validation
            - ask_user (bool): whether to ask user for configuration
            - verbose (int): level of verbosity, can be either 0, 1, or 2
            - plot (bool): whether to include plots for visual evaluations

        """
        logger.info("**" + "-" * 160 + "**")
        self.cv = cv
        self.ask_user = ask_user
        self.plot = plot
        self.verbose = verbose
        logger.info(("<Pipeline Setup. Config: %s-Fold Cross-Validation; "
                     "Do %s Ask User for Model and Metrics; "
                     "Level of Verbosity: %s; "
                     "Do %s Include Plots for Evaluations>") %
                    (self.cv, ["NOT", ""][int(self.ask_user)], self.verbose,
                     ["NOT", ""][int(self.plot)]))

        self.X_train, self.X_test, self.y_train, _ = load_features()
        self.baseline = self.y_train.mean()
        logger.info("\tFinished reading processed data.")

        for model, params_dict in self.GRID_SEARCH_PARAMS.items():
            if 'max_features' in params_dict:
                params_dict['max_features'] = range(1, self.X_train.shape[1], 2)

        # Get model, metrics and their names
        self.model_index = None
        self.metrics_index = None
        self.model_name = None
        self.metrics_name, self.metrics = None, None

        # Set up benchmark model and store benchmark scores
        self.benchmark = None
        self.benchmark_scores = [None] * len(self.METRICS)

        # Get default setting and hyperparameter grids
        self.default_args = None
        self.hyper_args, self.hyper_grids = None, None
        self.hyper_grid_index = 0
        self.best_args = None

        # Set up classifier and cross-validation iterators
        self.clf = None
        self.skf = StratifiedKFold(n_splits=self.cv, random_state=self.SEED,
                                   shuffle=True)
        logger.info(("\t%s-fold cross-validation generator set up with "
                     "random seed '%s'.") % (self.cv, self.SEED))

    def configure_model(self, model_index):
        """
        Configure model of the pipeline with indices from user input or
        automate generation. Meanwhile set default arguments and define the
        hyperparameter grids for the classifier.

        Inputs:
            - model_index (int): index of the model for the pipeline

        """
        logger.info("**" + "-" * 160 + "**\n\n")
        logger.info("**" + "-" * 160 + "**")

        # Set model index and model name
        self.model_index = model_index
        self.model_name = self.MODEL_NAMES[model_index]
        logger.info("<Model Configured. %s.>" % self.model_name)

        # Get model default arguments and hyperparameter grids
        self.default_args = self.DEFAULT_ARGS[self.model_name]

        params_grid = self.GRID_SEARCH_PARAMS[self.model_name]
        self.hyper_args = params_grid.keys()

        hyper_grids = itertools.product(*(params_grid[hyper_arg]
                                          for hyper_arg in self.hyper_args))
        self.hyper_grids = [tuple(map(lambda x: round(x, 3)
                                                if isinstance(x, float) else x,
                                      hyper_grid))
                            for hyper_grid in hyper_grids]

        # Set up model with default arguments
        self.clf = self.MODELS[model_index](**self.default_args)

    def configure_metrics(self, metrics_index):
        """
        Configure metrics of the pipeline with indices from user input or
        automate generation.

        Inputs:
            - metrics_index (int): index of the metrics for the pipeline

        """
        logger.info("\n\n#" + "-" * 130 + "#")
        # Set metrics index and metrics name
        self.metrics_index = metrics_index
        self.metrics_name = self.METRICS_NAMES[metrics_index]
        logger.info("<Metrics Configured. %s.>" % self.metrics_name.title())

        # Set up the metrics
        self.metrics = self.METRICS[metrics_index]

    def save_prob(self, predicted_prob, file_path, error=False):
        """
        Save predicted probabilities of a fitted model on the validation
        set to the corresponding log directory.

        Inputs:
            - predicted_prob (array): predicted probabilities on the validation
                set.
            - file_path (str): file name consists of directory, model index,
                and hyperparameter set index.

        """
        if not os.path.isfile(file_path):
            np.save(file_path, predicted_prob, allow_pickle=False)

            if self.verbose > 1:
                logger.info(("\n\t(Fitted Model: %s; Hyperparameter Grid %s) "
                             "Predicted probabilities saved to '%s'.") %
                            (self.model_name, self.hyper_grid_index,
                             file_path))

    def set_benchmark(self):
        """
        Train a default Decision Tree as the benchmark, and report its
        performance. Or just load relevant information and print.

        """
        logger.info("\n\nStart to set benchmark for the metrics %s." %
                    self.metrics_name.title())

        if not self.benchmark:
            self.benchmark = DecisionTreeClassifier(
                **self.DEFAULT_ARGS["Decision Tree"])
            self.benchmark.fit(self.X_train, self.y_train)

        if not self.benchmark_scores[self.metrics_index]:
            predicted_prob = self.cross_validation_prob()
            if self.metrics_name.startswith("Precision at"):
                self.benchmark_scores[self.metrics_index] = self.metrics(
                    self.y_train, predicted_prob)
            else:
                labels = [1 if prob >= 0.5 else 0 for prob in predicted_prob]
                self.benchmark_scores[self.metrics_index] = self.metrics(
                    self.y_train, labels)

        logger.info(("\t %s of the benchmark default Decision Tree model "
                     "is %4.3f.") % (self.metrics_name.title(),
                                     self.benchmark_scores[self.metrics_index]))

    def clf_predict_prob(self, X_val):
        """
        Apply the fitted classifier on validation sets and get predicted
        probabilities for each observation. For classifiers that cannot
        provide predicted probabilities, get its standardized decision
        function.

        Returns:
            (array of floats) predicted probabilities

        """
        if hasattr(self.clf, "predict_proba"):
            predicted_prob = self.clf.predict_proba(X_val)[:, 1]
        else:
            prob = self.clf.decision_function(X_val)
            predicted_prob = (prob - prob.min()) / (prob.max() - prob.min())

        return predicted_prob

    def cross_validation_prob(self):
        """
        With a classifier that has default and hyperparameters set, make
        cross-validation predictions on the training set and returns predicted
        probabilities of training observations being positive.

        Returns:
            (list of floats) predicted probabilities on the training set

        """
        cv_prob = np.zeros(self.y_train.shape[0])

        for train, val in self.skf.split(self.X_train, self.y_train):
            X_tr, X_val = self.X_train[train], self.X_train[val]
            y_tr = self.y_train[train]

            try:
                self.clf.fit(X_tr, y_tr)
                cv_prob[val] = self.clf_predict_prob(X_val)
            except:
                return None

        return cv_prob

    def get_predicted_prob(self, hyper_params):
        """
        Given a grid of hyperparameters, fit a model to get the predicted
        probabilities, or retrieve them from saved NumPy arrays.

        Inputs:
            - hyper_params ({str: float}): dict of a specific hyperparamter set

        Returns:
            (array of floats) predicted probabilities of the specific
                hyperparameter set
        """
        dir_path = PREDICTED_PROBS_DIR + self.model_name + "/"
        create_dirs(dir_path)
        prob_path = dir_path + '%s.npy' % self.hyper_grid_index

        if not os.path.isfile(prob_path):
            self.clf.set_params(**hyper_params)
            predicted_prob = self.cross_validation_prob()
            if not isinstance(predicted_prob, np.ndarray):
                return None

            self.save_prob(predicted_prob, prob_path)
        else:
            predicted_prob = np.load(prob_path)
            if self.verbose > 1:
                logger.info(("\t\t%s already fitted with hyperparameter set "
                             "%s. Predicted probabilities retrieved." %
                             (self.model_name, self.hyper_grid_index)))

        return predicted_prob

    def find_best_thresholds(self, predicted_prob):
        """
        Make predictions at threshold k, where k is the percentage of
        population at the highest probabilities to be classified as "positive".
        Evaluate the predictions and find the best threshold to use for the
        specific set of hyperparameters.

        Inputs:
            - predicted_prob (list of floats): predicted probabilities of
                the training set

        Returns:
            - (list of ints): predicted labels for the test/validation set
            - (float) best score the classifier achieves with the current set of
                hyperparameters

        """
        grid_best_score = 0
        grid_best_thresholds = []

        for k in self.DECISION_THRESHOLDS:
            labels = [1 if prob >= k else 0 for prob in predicted_prob]
            score = self.metrics(self.y_train, labels)

            if score >= grid_best_score and score != 0:
                grid_best_score = score
                grid_best_thresholds.append(k)

        return grid_best_thresholds, grid_best_score

    def tune(self):
        """
        Tune the classifier with certain metrics. Grid search through all the
        possible combinations of hyperparameters and try different thresholds,
        where each is the percentage of population at the highest probabilities
        to be classified as "positive". Find the best sets of hyperparameters
        and thresholds configuration.

        Returns:
            config ({tuple: list}): mapping hyperparameter values to population
                thresholds

        """
        logger.info(("\n\n<%s to be optimized on %s> Search Starts (%s of "
                     "hyperparameter sets):") % (self.model_name,
                                                 self.metrics_name.title(),
                                                 len(self.hyper_grids)))

        best_score = 0
        best_config = dict()
        for hyper_grid in self.hyper_grids:
            hyper_params = dict(zip(self.hyper_args, hyper_grid))

            predicted_prob = self.get_predicted_prob(hyper_params)
            if not isinstance(predicted_prob, np.ndarray):
                if self.verbose >= 1:
                    logger.info(("\tSet %s --- (Parameters: %s) HYPERPARAMETER "
                                 "SET NOT ALLOWED OR MODEL CANNOT CONVERGE. "
                                 "ABORTED.") % (self.hyper_grid_index,
                                                hyper_params))
                self.hyper_grid_index += 1
                continue

            grid_thresholds, score = self.find_best_thresholds(predicted_prob)
            if not grid_thresholds:
                continue

            if self.verbose >= 1:
                logger.info(("\tSet %s --- (Parameters: %s) Cross-validation %s"
                             " of %4.3f at the threshold(s) %s.") %
                            (self.hyper_grid_index, hyper_params,
                             self.metrics_name, score, grid_thresholds))

            if score >= best_score:
                if score > best_score:
                    best_config = dict()
                best_score = score
                best_config[hyper_grid] = (self.hyper_grid_index,
                                           grid_thresholds)

            self.hyper_grid_index += 1

        self.hyper_grid_index = 0

        logger.info(("<%s to be optimized on %s> Search Finished. The highest "
                     "cross-validation %s is %4.4f. There are %s best sets.") %
                    (self.model_name, self.metrics_name.title(),
                     self.metrics_name, best_score, len(best_config)))
        logger.info("\n#" + "-" * 130 + "#\n")

        return best_config

    def plot_model(self, hyper_params):
        """
        Plot the tuned models in to visually evaluate it in terms of predicted
        probabilities, precision-recall-population thresholds, area under curve,
        and feature importances. Images saved under "../logs/train/viz/".

        Inputs:
            - hyper_params ({str: float}): dict of a specific hyperparamter set

        """
        dir_path = VIZ_DIR + self.metrics_name + "/" + self.model_name + "/"
        create_dirs(dir_path)

        cv_prob = self.get_predicted_prob(hyper_params)
        title = "%s-%s-%s" % (self.model_name, self.metrics_name.title(),
                              self.hyper_grid_index)

        plot_predicted_scores(cv_prob, dir_path, title)

        plot_precision_recall(self.y_train, cv_prob, self.baseline, dir_path,
                              title)

        self.clf.set_params(**hyper_params)
        data = train_test_split(self.X_train, self.y_train, test_size=0.25,
                                random_state=self.SEED)
        plot_auc_roc(self.clf, *data, dir_path, title)

        if hasattr(self.clf, "feature_importances_"):
            importances = self.clf.feature_importances_
            col_names = read_feature_names(INPUT_DIR, 'train_features.txt')
            plot_feature_importances(importances, col_names, dir_path, top_n=5,
                                     title=title)

    def predict(self):
        """
        Fit a tuned model on the training set and make predictions on the test
        set.

        Inputs:
            - hyper_params ({str: float}): dict of a specific hyperparamter set

        Return:
            - train_time: training time in seconds
            - test_time: test time in seconds
            - labels: predicted labels on the test set

        """
        train_start = time.time()
        try:
            self.clf.fit(self.X_train, self.y_train)
        except:
            return None, None, None
        train_time = time.time() - train_start

        test_start = time.time()
        predicted_prob = self.clf_predict_prob(self.X_test)
        test_time = time.time() - test_start

        return train_time, test_time, predicted_prob

    def write_results(self, predicted_prob, thresholds):
        """
        Write results ready for submission.

        Inputs:
            - predicted_prob (list of floats): predicted probabilities of
                the test set
            - thresholds (list of ints): corresponding population thresholds for
                best performances

        """
        dir_path = PREDICTIONS_DIR + self.metrics_name + "/" \
                   + self.model_name + "/"
        create_dirs(dir_path)
        template = pd.read_csv(RAW_DATA_DIR + "gender_submission.csv")

        for k in thresholds:
            file_name = "%s-%s-%s.csv" % (self.model_name,
                                          self.hyper_grid_index, k)
            labels = [1 if prob >= k else 0 for prob in predicted_prob]

            copy = pd.DataFrame({'PassengerId': template.PassengerId,
                                 'Survived': labels})
            copy.to_csv(dir_path + file_name, index=False)

    def train_eval(self):
        """


        """
        self.set_benchmark()
        best_config = self.tune()

        for hyper_grid, (set_index, thresholds) in best_config.items():
            self.hyper_grid_index = set_index
            hyper_params = dict(zip(self.hyper_args, hyper_grid))
            self.clf.set_params(**hyper_params)

            if self.plot:
                self.plot_model(hyper_params)

            tr_time, ts_time, predicted_prob = self.predict()
            if predicted_prob is None:
                continue

            self.write_results(predicted_prob, thresholds)

        self.hyper_grid_index = 0

    def run(self):
        """


        """
        if self.ask_user:
            model_index = ask(self.MODEL_NAMES, "model")
            self.configure_model(model_index)

            metrics_index = ask(self.METRICS_NAMES, "metrics")
            self.configure_metrics(metrics_index)

            self.train_eval()
        else:
            for i in range(len(self.MODELS)):
                self.configure_model(i)

                for j in range(len(self.METRICS)):
                    self.configure_metrics(j)

                    self.train_eval()


# ----------------------------------------------------------------------------#
if __name__ == "__main__":

    desc = ("Build a machine learning pipeline, set the benchmark with a "
            "default decision tree, build the model with one or all of the "
            "three supervised learning algorithms, tune hyperparameters with "
            "cross-validation, evaluate and make predictions on the test data.")
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--start_clean', dest='start_clean', type=int, default=0,
                        help=("There can be predicted probabilities "
                              "visualizations, and predictions in the folder "
                              "from previous runs, which the pipeline can pick "
                              "up and expedite the current training and "
                              "evaluation. Please indicate if you would like "
                              "it to start from scratch instead (1 for true or "
                              "0 for false)."))
    parser.add_argument('--ask', dest='ask_user', type=int, default=1,
                       help=("Please specify whether the script should ask for "
                             "user configuration to run on one model-metrics "
                             "pair or all of them (1 for true or 0 for false)."))
    parser.add_argument('--verbose', dest='verbose', type=int, default=1,
                        help="Please specify the level of verbosity (0, 1, 2).")
    parser.add_argument('--plot', dest='plot', type=int, default=1,
                        help=("Please specify whether to include visualized "
                              "evaluations of the best models (1 for true or "
                              "0 for false)."))

    args = parser.parse_args()

    if args.start_clean:
        start_clean()
    train_pipe = ModelingPipeline(cv=5, ask_user=bool(args.ask_user),
                                  verbose=args.verbose, plot=bool(args.plot))
    train_pipe.run()

    _ = input("Press any key to exit.")
