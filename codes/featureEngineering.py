"""
Title:       Build a preprocessing pipeline that helps user preprocess training
             and test data from the corresponding CSV input files.

Description: Fill in missing values, discretize continuous variables, generate
             new features, deal with categorical variables with multiple levels,
             scale data, and save preprocessed data.

Author:      Kunyu He, CAPP'20, The University of Chicago

"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler

#----------------------------------------------------------------------------#
INPUT_DIR = "../data/"
OUTPUT_DIR = "../processed_data/"
LOG_DIR = "../logs/featureEngineering/"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TRAIN_FEATURES_FILE = 'train_features.txt'
TEST_FEATURES_FILE = 'test_features.txt'

# logging
logger= logging.getLogger('featureEngineering')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)
fh = logging.FileHandler(LOG_DIR + time.strftime("%Y%m%d-%H%M%S") + '.log')
logger.addHandler(fh)

pd.set_option('mode.chained_assignment', None)


#----------------------------------------------------------------------------#
def read_data(file_name, drop_na=False):
    """
    Read credit data in the .csv file and data types from the .json file.

    Inputs:
        - data_file (string): name of the data file.
        - drop_na (bool): whether to drop rows with any missing values

    Returns:
        (DataFrame) clean data set with correct data types

    """
    data = pd.read_csv(INPUT_DIR + file_name)

    if drop_na:
        data.dropna(axis=0, inplace=True)

    return data


def ask(names, message):
    """
    Ask user for their choice of index for model or metrics.

    Inputs:
        - name (list of strings): name of choices
        - message (str): type of index to request from user

    Returns:
        (int) index for either model or metrics
    """
    indices = []

    print("\nUp till now we support:")
    for i, name in enumerate(names):
        print("%s. %s" % (i + 1, name))
        indices.append(str(i + 1))

    index = input("Please input a %s index:\n" % message)

    if index in indices:
        return int(index) - 1
    else:
        print("Input wrong. Type one in {} and hit Enter.".format(indices))
        return ask(names, message)


def read_feature_names(dir_path, file_name):
    """
    Read .txt files with only one line as feature names separated by ",". Save
    the output to a list of feature names.

    Returns:
        (list of strings) list of feature names.
    """
    with open(dir_path + file_name, 'r') as handle:
        return np.array(handle.readline().split(","))


class FeaturePipeLine:
    """
    Preprocess pipeline for a data set from CSV file. Modify the class
    variables to fill in missing values, combine multinomial variables to ones
    with less levels and binaries, and apply one-hot-encoding. Then split data
    into features and target, drop rows with missing labels and some columns.
    At last, apply scaling.

    """
    TO_FILL_CON = {'Age', 'Fare'}

    TO_FILL_OBJ = {'Cabin': "NO",
                   'Embarked': None}

    TO_EXTRACT_OBJ = {'Name': ('Title', r' ([A-Za-z]+)\.')}

    TO_CREATE_CON = {'FamilySize': (['SibSp', 'Parch'], lambda x, y: x + y + 1),
                     'IsAlone': (['FamilySize'], lambda x: np.where(x >= 1,
                                                                    1, 0))}

    TO_DISCRETIZE = {'Age': (False, 5, 0),
                     'Fare': (True, 4, 3)}

    TO_COMBINE = {'Cabin': None,
                  'Title': {"MISC": None}}

    TO_BINARIES = {'Sex': 'auto',
                   'Cabin': 'auto',
                   'IsAlone': 'auto'}

    TO_ONE_HOT = ['Pclass', 'Embarked', 'Age', 'Title']

    TARGET = 'Survived'
    TO_DROP = ['PassengerId', 'Ticket', 'Name', 'SibSp', 'Parch']

    SCALERS = [StandardScaler, MinMaxScaler]
    SCALER_NAMES = ["Standard Scaler", "MinMax Scaler"]

    def __init__(self, file_name, ask_user=True, verbose=True,
                 drop_na=False, test=False):
        """
        Construct a preprocessing pipeline given name of the data file.

        Inputs:
            - file_name (string): name of the data file
            - ask_user (bool): whether to ask user for configuration
            - verbose (bool): whether to make extended printing in
                preprocessing
            - drop_na (bool): whether to drop rows with missing values
            - test (bool): whether this is a pipeline built on test data

        """
        logger.info("**" + "-" * 140 + "**")
        logger.info("Creating the preprocessing pipeline for '%s'." %
                    file_name)
        self.data = read_data(file_name, drop_na)
        self.verbose = verbose
        self.test = test
        logger.info("\tFinished reading cleaned data.\n")

        if not self.test:
            if ask_user:
                self.scaler_index = ask(self.SCALER_NAMES, "scaler")
            else:
                self.scaler_index = 0
            self.scaler = self.SCALERS[self.scaler_index]()
            logger.info("<Training data preprocessing> Pipeline using %s." %
                        (self.SCALER_NAMES[self.scaler_index]))
        else:
            self.scaler = joblib.load(INPUT_DIR + 'fitted_scaler.pkl')
            logger.info("<Test data preprocessing> Pre-fitted scaler loaded.")

        self.X = None
        self.y = None

    def con_fill_na(self):
        """
        Take the continuous variables, impute the missing features with column
        medians.

        Returns:
            (self) pipeline with missing values in the numerical columns imputed

        """
        logger.info("\n\nStart to impute missing values continuous variables:")

        for var in self.TO_FILL_CON:
            imputed = self.data[var].median()
            self.data[var] = self.data[var].fillna(imputed)

            if self.verbose:
                logger.info(("\tMissing values in '%s' imputed with column "
                             " median %4.3f.") % (var, imputed))

        return self

    def str_fill_na(self):
        """
        Fill in missing data with desired string entry.

        Returns:
            (self) pipeline with missing values in the object columns filled.

        """
        logger.info("\n\nStart to fill in missing values:")

        for var, fill in self.TO_FILL_OBJ.items():
            # if no value is provided to fill in, use the most frequent one
            if fill is None:
                fill = self.data[var].mode()[0]
            self.data[var].fillna(value=fill, inplace=True)

            if self.verbose:
                logger.info("\tFilled missing values in '%s' with '%s'." %
                            (var, fill))

    def discretize(self):
        """
        Discretizes continuous variables into multinomials.

        Returns:
            (self) pipeline with some numerical columns discretized.

        """
        logger.info("\n\nStart to discretize continuous variables:")

        for var, (qcut, bins, precision) in self.TO_DISCRETIZE.items():
            if qcut:
                self.data[var] = pd.qcut(self.data[var], bins,
                                         precision=precision).cat.codes
            else:
                self.data[var] = pd.cut(self.data[var], bins,
                                        precision=precision).cat.codes

            if self.verbose:
                if isinstance(bins, list):
                    bins = len(bins) - 1

                info = "\tDiscretized '%s' into %s %s-sized buckets."
                if self.data[var].isnull().sum() > 0:
                    info += " Here '-1' indicates that the value is missing."
                logger.info(info % (var, bins,
                                    ['differently', 'equally'][int(qcut)]))

            if bins > 2:
                self.TO_ONE_HOT.append(var)

        return self

    def str_extract(self):
        """
        Extract information from object columns with regular expression.

        Returns:
            (self) pipeline with new object columns extracted from existing ones

        """
        logger.info("\n\nStart to extract information from string variables:")

        for var, (new_var, reg_exp) in self.TO_EXTRACT_OBJ.items():
            self.data[new_var] = self.data[var].str.extract(reg_exp,
                                                            expand=False)

            if self.verbose:
                logger.info("\t'%s' extracted from '%s'." % (new_var, var))

        return self

    def con_create(self):
        """
        Create continuous variables with lambda functions on existing numerical
        columns.

        """
        logger.info("\n\nStart to create new continuous variables.")
        for new_var, (variables, func) in self.TO_CREATE_CON.items():
            self.data[new_var] = func(*[self.data[var] for var in variables])

            if self.verbose:
                logger.info("\tFunction applied on variables %s to create '%s'"
                            % (variables, new_var))

    def to_combine(self):
        """
        Combine some unnecessary levels of multinomials.

        Returns:
            (self) pipeline with less frequent levels in the multinomial columns
                combined.

        """
        logger.info("\n\nStart to combine unnecessary levels of multinomials.")

        for var, dict_combine in self.TO_COMBINE.items():
            if not dict_combine:
                dict_combine = {"YES": [val for val in self.data[var].unique()
                                        if val != "NO"]}

            for combined, lst_combine in dict_combine.items():
                if not lst_combine:
                    freqs = self.data[var].value_counts(normalize=True)
                    lst_combine = freqs[freqs < 0.05].index
                self.data.loc[self.data[var].isin(lst_combine), var] = combined

            if self.verbose:
                logger.info("\tCombinations of levels on '%s'." % var)

        return self

    def to_binary(self):
        """
        Transform variables to binaries.

        Returns:
            (self) pipeline with chosen columns transformed to binaries.

        """
        logger.info(("\n\nFinished transforming the following variables: %s to "
                     "binaries.") % (list(self.TO_BINARIES.keys())))

        for var, cats in self.TO_BINARIES.items():
            enc = OrdinalEncoder(categories=cats)
            self.data[var] = enc.fit_transform(np.array(self.data[var]).\
                                               reshape(-1, 1))

        return self

    def one_hot(self):
        """
        Creates binary/dummy variables from multinomials, drops the original
        and inserts the dummies back.

        Returns:
            (self) pipeline with one-hot-encoding applied on categorical vars.

        """
        logger.info(("\n\nFinished applying one-hot-encoding to the following "
                     "categorical variables: %s\n\n") % self.TO_ONE_HOT)

        for var in self.TO_ONE_HOT:
            dummies = pd.get_dummies(self.data[var], prefix=var)
            self.data.drop(var, axis=1, inplace=True)
            self.data = pd.concat([self.data, dummies], axis=1)

        return self

    def feature_target_split(self):
        """
        Drop rows with missing labels, drop some columns that are not relevant
        or have too many missing values, split the features (X) and target (y)
        Write columns names to "feature_names.txt" in the output directory.

        """
        if not self.test:
            self.data.dropna(axis=0, subset=[self.TARGET], inplace=True)
            self.y = self.data[self.TARGET]
            self.data.drop(self.TARGET, axis=1, inplace=True)
            logger.info("Finished extracting the target (y).")

        self.data.drop(self.TO_DROP, axis=1, inplace=True)
        self.data.dropna(axis=0, inplace=True)
        self.X = self.data
        logger.info("Finished extracting the features (X).")

        if not self.test:
            file_name = TRAIN_FEATURES_FILE
        else:
            file_name = TEST_FEATURES_FILE

        with open(OUTPUT_DIR + file_name, 'w') as file:
            file.write(",".join(self.X.columns))
            logger.info("\t%s feature names wrote to '%s' under directory '%s'"
                        % (["Train", "Test"][int(self.test)], file_name,
                           OUTPUT_DIR))

    def compare_train_test(self):
        """
        Compare the features in the training and test set after preprocessing.
        For those in the training set but not the test set, insert a column with
        all zeros at the same column index in the test set. For those in the
        test set but are not in the training set, drop them from the test set.

        """
        train_features = read_feature_names(OUTPUT_DIR, 'train_features.txt')
        test_features = read_feature_names(OUTPUT_DIR, 'test_features.txt')

        to_drop = [var for var in test_features if var not in train_features]
        self.data.drop(to_drop, axis=1, inplace=True)
        logger.info(("\n\n%s are in the test set but are not in the training "
                     "set, dropped from the test set.") % to_drop)

        to_add = [(i, var) for (i, var) in enumerate(train_features)
                  if var not in test_features]
        logger.info(("Start to add those are in the training set but not the "
                     "test set to the test set:"))
        for i, var in to_add:
            self.data.insert(loc=i, column=var, value=0)
            if self.verbose:
                logger.info("\t'%s' added to the %sth column of the test set "
                            "with all zeros." % (var, i))

    def scale(self):
        """
        Fit and transform the scaler on the training data and return the
        scaler data to scale test data.

        Returns:
            (self) pipeline with scaled data. The fitted scaler dumped to
                "../data/"

        """
        logger.info("\n")

        if not self.test:
            self.scaler.fit(self.X.values.astype(float))
            joblib.dump(self.scaler, INPUT_DIR + 'fitted_scaler.pkl')
            logger.info(("<Training data preprocessing> Fitted scaler dumped "
                         "to '%s' under directory '%s'.") % ('fitted_scaler.pkl',
                                                             INPUT_DIR))

        self.X = self.scaler.transform(self.X.values.astype(float))
        logger.info("Finished scaling the feature matrix.")

        return self

    def save_data(self):
        """
        Saves the feature matrix and target as numpy arrays in the output
        directory.

        """
        extension = ["_train.npy", "_test.npy"][int(self.test)]

        np.save(OUTPUT_DIR + "X" + extension, self.X, allow_pickle=False)
        if not self.test:
            np.save(OUTPUT_DIR + "y" + extension, self.y.values.astype(float),
                    allow_pickle=False)

        logger.info(("\n\nSaved the resulting NumPy matrices to directory '%s'. "
                     "Features are in 'X%s' and target is in 'y%s'.") %
                     (OUTPUT_DIR, extension, extension))

    def preprocess(self):
        """
        Finish preprocessing the data file.

        """
        self.con_fill_na().str_fill_na()
        self.discretize().str_extract().con_create()
        self.to_combine().to_binary().one_hot().feature_target_split()
        if self.test:
            self.compare_train_test()
        self.scale().save_data()

        logger.info("\n\n<Finished processing %s data>" % \
                    (["training", "test"][int(self.test)]))
        logger.info("**" + "-" * 140 + "**\n\n")


#----------------------------------------------------------------------------#
if __name__ == "__main__":
    desc = ("Build a preprocessing pipeline that helps user preprocess "
            "training and test data from the corresponding CSV input files. "
            "Fill in missing values, discretize continuous variables, generate "
            "new features, deal with categorical variables with multiple "
            "levels, scale data, and save preprocessed data.")
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--ask', dest='ask_user', type=int, default=0,
                        help=(
                            "Please specify whether the script should ask for "
                            "user configuration to run on one model-metrics "
                            "pair or all of them (1 for true or 0 for false)."))
    parser.add_argument('--verbose', dest='verbose', type=int, default=1,
                        help=("Please specify whether the pipeline should be "
                              "verbose (1 for true or 0 for false)."))
    parser.add_argument('--drop_na', dest='drop_na', type=int, default=0,
                        help=("Please specify whether to drop the missing "
                              "values when reading input data.(1 for true or 0 "
                              "for false)."))
    args = parser.parse_args()
    args_dict = {'ask_user': bool(args.ask_user),
                 'verbose': bool(args.verbose),
                 'drop_na': bool(args.drop_na)}

    training_pipeline = FeaturePipeLine(TRAIN_FILE, **args_dict, test=False)
    training_pipeline.preprocess()

    test_pipeline = FeaturePipeLine(TEST_FILE, **args_dict, test=True)
    test_pipeline.preprocess()
