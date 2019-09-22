# ML-Pipeline-for-Titanic-Challenge

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f494ae7e53a64bb2ad7fd826d137a6f7)](https://www.codacy.com/app/kunyuhe/ML-Pipeline-for-Titanic-Challenge?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=KunyuHe/ML-Pipeline-for-Titanic-Challenge&amp;utm_campaign=Badge_Grade)<a href="https://codeclimate.com/github/KunyuHe/ML-Pipeline-for-Titanic-Challenge/maintainability"><img src="https://api.codeclimate.com/v1/badges/bd9c52f4654c13c8fce8/maintainability" /></a> [![Documentation Status](https://readthedocs.org/projects/pydocstyle/badge/?version=stable)](http://www.pydocstyle.org/en/stable/?badge=stable)



## 1. Introduction

> The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.



> One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.



In this repository, we performed an analysis of what sorts of people were likely to survive in the Titanic shipwreck. In particular, the task is to build a machine learning pipeline that predicts whether a passenger would survive the tragedy.



The pipeline has four components:

1. Read Data
2. Generate Features/Predictors
3. Build Classifier
4. Evaluate Classifier



As you might have noticed, there is no exploratory data analysis part in our pipeline. One reason for this is that there are already tons of well-established analysis on the data set and many enlightening findings from visualizations. We referred to [Harry Emeric's Kaggle kernel](https://www.kaggle.com/harryem/feature-engineering-on-the-titanic-for-0-81339) and [Andrew Conti's repo](https://github.com/agconti/kaggle-titanic) for EDA and tips in feature engineering.



## 2. Get Data

> - Output Directory: `../data/ `   *(All paths hereby would be relative to the `/codes/` directory)*



Data can be manually downloaded from [this link](https://www.kaggle.com/c/3136/download-all) on Kaggle . Data sets extracted from the downloaded file include:

- `train.csv`: the training set, used to build the machine learning models
- `test.csv`: the test set, used to see how well the model performs on unseen data
- `gender_submission.csv`: a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.



Below is a data dictionary for the training and test sets from the corresponding [Kaggle data page](https://www.kaggle.com/c/titanic/data):

| **Variable** | **Definition**                             | **Key**                                                      |
| ------------ | ------------------------------------------ | ------------------------------------------------------------ |
| survival     | Survival                                   | 0 = No, 1 = Yes                                              |
| pclass       | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                                    |
| sex          | Sex                                        |                                                              |
| Age          | Age in years                               | Age is fractional if less than 1.<br/>If the age is estimated, is it in the form of xx.5 |
| sibsp        | # of siblings / spouses aboard the Titanic | Sibling = brother, sister, stepbrother, stepsister<br/>Spouse = husband, wife (mistresses and fiancés were ignored) |
| parch        | # of parents / children aboard the Titanic | Parent = mother, father<br/>Child = daughter, son, stepdaughter, stepson<br/>Some children travelled only with a nanny, therefore parch=0 for them. |
| ticket       | Ticket number                              |                                                              |
| fare         | Passenger fare                             |                                                              |
| cabin        | Cabin number                               |                                                              |
| Embarked     | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton               |



## 3. Feature Engineering

> - Input Directory: `../data/`
> - Output Directory: `../processed_data/`
> - Logging Directory = `../logs/featureEngineering/`
> - Code Script: [featureEngineering.py](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/featureEngineering.py)
> - Test Script: *in progress*



For feature engineering, we created a preprocessing pipeline that helps user preprocess training and test data from the corresponding CSV input files. **Note that since training and test data come from different files, our preprocessing respects the independence between them.**



### 3.1 Impute Missing Values

There are many missing values in `Age`, which is expected to be crucial to our prediction. Apparently, we cannot simply drop the predictor itself or rows with missing values on it. Instead, imputed the missing values with column median and discretized `Age` into three buckets: *(0, 18], (18, 60], (60, +oo)*. Now `Age` becomes a categorical variable with no value missing.



We also noticed that there are a lot of passengers with missing `Cabin`. Combined with the context, we suppose that many passengers didn't have a cabin on the ship. Thus we assign "None" to fields with missing values and combined all other types of `Cabin` values as "Yes". By doing this, we turned `Cabin` into a binary indicator for whether a passenger has a cabin.

 

Another predictor `Embarked` also has some values missing, and likewise, we filled in with "Unknown". Hereby, there are no more values missing in our training data.



### 3.2 Other Preprocessing Steps



