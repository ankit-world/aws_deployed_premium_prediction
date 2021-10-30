from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from application_logging.logging import logger_app
import numpy as np
import warnings
from sklearn import *
import os

warnings.filterwarnings("ignore")


class Model_Finder:

    def __init__(self):
        try:
            self.logger_object = logger_app()
            # self.logFilepath = "Training_Logs/ModelTrainingLog.txt"
            self.logFilepath = os.path.join("Training_Logs", "ModelTrainingLog.txt")
            self.logfile = open(self.logFilepath, mode='a')
            self.lasso = Lasso()
            self.dt = DecisionTreeRegressor()
            self.rf = RandomForestRegressor()
            self.gb = GradientBoostingRegressor()
            self.logger_object.log(self.logfile, 'Instances have been created for all the models')
            self.logfile.close()
        except Exception as e:
            self.logger_object.log(self.logfile, log_massage="Exception : " + str(e))
            self.logfile.close()
            raise e

    def get_best_params_for_lasso_regression(self, x_train, y_train):
        """
                Method Name: get_best_params_for_lasso_regression
                Description: This method will load the Lasso Regression model and will tune the in-build parameter to
                get best possible accuracy.
                Output: None
                On Failure: Raise Exception


        """
        self.logfile = open(self.logFilepath, mode='a')
        self.logger_object.log(self.logfile, 'Entered the get_best_params_for_lasso_regression method of the '
                                             'Model_Finder class')

        try:

            # Creating an object of the lasso Regularization
            self.lassocv = LassoCV(alphas=None, cv=50, max_iter=200000, normalize=True)

            # finding the best parameter
            self.lassocv.fit(x_train, y_train)

            # Creating a new model with required parameter
            self.lasso = Lasso(alpha=self.lassocv.alpha_)

            # Training the new model
            self.lasso.fit(x_train, y_train)
            self.logger_object.log(self.logfile, 'Lasso Regression best params: ' + str(
                self.lassocv.alpha_) + '. Exited the get_best_params_for_linear_regression method of the Model_Finder '
                                       'class')
            self.logfile.close()
            return self.lasso

        except Exception as e:
            self.logger_object.log(self.logfile, 'Exception occurred in get_best_params_for_lasso_regression method '
                                                 'of the Model_Finder class.')
            self.logger_object.log(self.logfile, 'Lasso Repressor Parameter tuning  failed.Exited the '
                                                 'get_best_params_for_random_forest method of the Model_Finder class')
            self.logfile.close()
            raise Exception()

    def get_best_params_for_decision_tree(self, x_train, y_train):
        """
                Method Name: get_best_params_for_decision_tree
                Description: This method will load the decision tree model and will tune the in-build parameter to get
                best possible accuracy.
                Output: None
                On Failure: Raise Exception


        """
        self.logfile = open(self.logFilepath, mode='a')
        self.logger_object.log(self.logfile, 'Entered the get_best_params_for_decision_tree method of the '
                                             'Model_Finder class')

        try:
            self.tree_grid = {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],

                              "splitter": ["best", "random"],
                              "min_samples_split": [10, 20, 40],
                              "max_depth": [2, 6, 8],
                              "max_features": ["auto", "sqrt", "log2"],
                              "min_samples_leaf": [20, 40, 80],
                              "min_weight_fraction_leaf": [0.1],
                              "max_leaf_nodes": [5, 20, 100],
                              "ccp_alpha": np.random.rand(20)
                              }

            # Creating an object of the Random Search class
            self.dt_cv = RandomizedSearchCV(estimator=self.dt, param_distributions=self.tree_grid, n_iter=10, cv=5,
                                            verbose=2, random_state=100, n_jobs=-1)

            # finding the best parameter
            self.dt_cv.fit(x_train, y_train)

            # creating a new model with the best parameters and estimator
            self.dt = self.dt_cv.best_estimator_

            # Training the new model
            self.dt.fit(x_train, y_train)
            self.logger_object.log(self.logfile, 'Decision Trees best params: ' + str(
                self.dt_cv.best_params_) + '. Exited the get_best_params_for_decision_tree method of the Model_Finder '
                                           'class')

            self.logfile.close()
            return self.dt

        except Exception as e:
            self.logger_object.log(self.logfile,
                                   'Exception occurred in get_best_params_for_decision_tree '
                                   'method of the Model_Finder class.')
            self.logger_object.log(self.logfile,
                                   'Decision Tree Regressor Parameter tuning  failed. Exited the '
                                   'get_best_params_for_random_forest method of the Model_Finder class')
            self.logfile.close()
            raise Exception()

    def get_best_params_for_random_forest(self, x_train, y_train):
        """
                Method Name: get_best_params_for_random_forest
                Description: This method will load the Random Forest model and will tune the in-build parameter to get
                best possible accuracy.
                Output: None
                On Failure: Raise Exception


        """

        self.logfile = open(self.logFilepath, mode='a')
        self.logger_object.log(self.logfile,
                               'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:

            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt', 'log2']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 1000, 10)]
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10, 14]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4, 6, 8]
            # criterian
            criterian = ['squared_error', 'absolute_error', 'poisson']

            self.random_grid = {'n_estimators': n_estimators,
                                'max_features': max_features,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'criterion': criterian
                                }

            # Creating an object of the Grid Search class
            self.rf_cv = RandomizedSearchCV(estimator=self.rf, param_distributions=self.random_grid, n_iter=10, cv=2,
                                            verbose=2, random_state=100, n_jobs=-1)

            # finding the best parameters
            self.rf_cv.fit(x_train, y_train)

            # creating a new model with the best parameters and estimator
            self.rf = self.rf_cv.best_estimator_

            # training the mew model
            self.rf.fit(x_train, y_train)
            self.logger_object.log(self.logfile, 'Random Forest Regressor best params: ' + str(
                self.rf_cv.best_params_) + '. Exited the get_best_params_for_random_forest '
                                           'method of the Model_Finder class')
            self.logfile.close()
            return self.rf
        except Exception as e:
            self.logger_object.log(self.logfile,
                                   'Exception occurred in get_best_params_for_random_forest method of the '
                                   'Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.logfile,
                                   'Random Forest Regressor Parameter tuning  failed. Exited the '
                                   'get_best_params_for_random_forest method of the Model_Finder class')
            self.logfile.close()
            raise Exception()

    def get_best_params_for_gradient_boosting(self, x_train, y_train):
        """
                Method Name: get_best_params_for_gradient_boosting
                Description: This method will load the gradient boosting model and will tune the in-build parameter to
                get best possible accuracy.
                Output: None
                On Failure: Raise Exception


        """

        self.logfile = open(self.logFilepath, mode='a')
        self.logger_object.log(self.logfile,
                               'Entered the get_best_params_for_gradient_boosting method of the Model_Finder class')

        try:
            # initializing with different combination of parameters
            n_estimators = [int(x) for x in np.linspace(start=200, stop=4000, num=10)]
            self.gb_grid = {'learning_rate': [0.01, 0.02, 0.03, 0.04],
                            'subsample': [0.9, 0.5, 0.2, 0.1],
                            'n_estimators': n_estimators,
                            'max_depth': [4, 6, 8, 10],
                            'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                            'criterion': ['friedman_mse'],
                            'max_features': ['auto', 'sqrt', 'log2']
                            }
            # Creating an object of the Random Search class
            self.gb_cv = RandomizedSearchCV(estimator=self.gb, param_distributions=self.gb_grid,
                                            scoring='neg_mean_squared_error', n_iter=10, cv=2, verbose=2,
                                            random_state=300, n_jobs=-1)

            # finding the best parameters
            self.gb_cv.fit(x_train, y_train)

            # creating a new model with the best parameters and estimator
            self.gb = self.gb_cv.best_estimator_

            # training the mew model
            self.gb.fit(x_train, y_train)
            self.logger_object.log(self.logfile, 'Gradient Boosting Regressor best params: ' + str(
                self.gb_cv.best_params_) + '. Exited the get_best_params_for_gradient_boosting '
                                           'method of the Model_Finder class')
            self.logfile.close()
            return self.gb
        except Exception as e:
            self.logger_object.log(self.logfile,
                                   'Exception occurred in get_best_params_for_random_forest method of the '
                                   'Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.logfile,
                                   'Gradient Boosting Regressor Parameter tuning  failed. Exited the '
                                   'get_best_params_for_gradient_boosting method of the Model_Finder class')
            self.logfile.close()
            raise Exception()

    def get_best_model(self, x_train, x_test, y_train, y_test):
        """
                Method Name: get_best_params_for_gradient_boosting
                Description: This method will compare the accuracies of all the models and the model with highest
                accuracy will be picked.
                Output: None
                On Failure: Raise Exception


        """

        self.logfile = open(self.logFilepath, mode='a')
        self.logger_object.log(self.logfile, 'Entered the get_best_model method of the Model_Finder class')
        self.list_of_files = []
        self.list_of_files_score = []

        try:

            # create best model for linear regression
            self.lasso = self.get_best_params_for_lasso_regression(x_train, y_train)
            self.prediction_lasso = self.lasso.predict(x_test)

            # Train Accuracy of Lasso Regression
            self.lasso_score_train = self.lasso.score(x_train, y_train)
            # print('lasso Train accuracy ', str(self.lasso_score_train))

            # Test Accuracy of Lasso Regression
            self.lasso_score_test = metrics.r2_score(y_test, self.prediction_lasso)
            # print('lasso Test accuracy ', str(self.lasso_score_test))
            self.list_of_files_score.append(self.lasso_score_test)

            # Difference in train and test accuracy
            self.diff_lasso = abs(self.lasso_score_train - self.lasso_score_test)
            # print('lasso accuracy train and test difference ', str(self.diff_lasso))
            self.list_of_files.append(self.diff_lasso)
            self.logfile = open(self.logFilepath, mode='a')
            self.logger_object.log(self.logfile,
                                   'Test Accuracy for Lasso Regression Model :' + str(self.lasso_score_test))
            self.logger_object.log(self.logfile,
                                   'Train and Test Accuracy difference for Lasso Regression Model :' + str(
                                       self.diff_lasso))
            self.logfile.close()

            # create best model for Decision Tree
            self.dt = self.get_best_params_for_decision_tree(x_train, y_train)
            self.prediction_dt = self.dt.predict(x_test)

            # Train Accuracy of Lasso Regression
            self.dt_score_train = self.dt.score(x_train, y_train)
            # print('decision tree Train accuracy', str(self.dt_score_train))

            # Test Accuracy of Decision Tree
            self.dt_score_test = metrics.r2_score(y_test, self.prediction_dt)
            # print('decision tree Test accuracy', str(self.dt_score_test))
            self.list_of_files_score.append(self.dt_score_test)

            # Difference in train and test accuracy
            self.diff_dt = abs(self.dt_score_train - self.dt_score_test)
            # print('Decision Tree accuracy train and test difference ', str(self.diff_dt))
            self.list_of_files.append(self.diff_dt)

            self.logfile = open(self.logFilepath, mode='a')
            self.logger_object.log(self.logfile, 'Accuracy for Decision Tree Model :' + str(self.dt_score_test))
            self.logger_object.log(self.logfile,
                                   ' Train and Test Accuracy  Difference for Decision Tree Model :' + str(self.diff_dt))
            self.logfile.close()

            # Create best model for Random Forest
            self.rf = self.get_best_params_for_random_forest(x_train, y_train)
            self.prediction_rf = self.rf.predict(x_test)

            # Train Accuracy of Random Forest
            self.rf_score_train = self.rf.score(x_train, y_train)
            # print('Random Forest Train Accuracy', str(self.rf_score_train))

            # Test Accuracy of Random Forest
            self.rf_score_test = metrics.r2_score(y_test, self.prediction_rf)
            # print('Random Forest Test Accuracy', str(self.rf_score_test))
            self.list_of_files_score.append(self.rf_score_test)

            # Difference in Train and Test Accuracy
            self.diff_rf = abs(self.rf_score_train - self.rf_score_test)
            # print('Random Forest accuracy train and test difference ', str(self.diff_rf))
            self.list_of_files.append(self.diff_rf)

            self.logfile = open(self.logFilepath, mode='a')
            self.logger_object.log(self.logfile, ' Test Accuracy for Random Forest Model :' + str(self.rf_score_test))
            self.logger_object.log(self.logfile,
                                   ' Train and Test Accuracy Difference for Random Forest Model :' + str(self.diff_rf))
            self.logfile.close()

            # Create best model for Gradient Boosting
            self.gb = self.get_best_params_for_gradient_boosting(x_train, y_train)
            self.prediction_gb = self.gb.predict(x_test)

            # Train Accuracy of Gradient Boosting
            self.gb_Score_train = self.gb.score(x_train, y_train)
            # print('gradient boosting Train Accuracy', str(self.gb_Score_train))

            # Test Accuracy of Gradient Boosting
            self.gb_score_test = metrics.r2_score(y_test, self.prediction_gb)
            # print('gradient boosting Test Accuracy', str(self.gb_score_test))
            self.list_of_files_score.append(self.gb_score_test)

            # Difference in Train and Test Accuracy
            self.diff_gb = abs(self.gb_Score_train - self.gb_score_test)
            # print('Gradient Boosting accuracy train and test difference ', str(self.diff_gb))
            self.list_of_files.append(self.diff_gb)

            self.logfile = open(self.logFilepath, mode='a')
            self.logger_object.log(self.logfile,
                                   'Test Accuracy for Gradient Boosting Model :' + str(self.gb_score_test))
            self.logger_object.log(self.logfile,
                                   ' Train and Test Accuracy difference for Gradient Boosting Model :' + str(
                                       self.diff_gb))
            self.logfile.close()

            # Keeping model with highest accuracy
            maxi = max(self.list_of_files_score)
            if maxi == self.lasso_score_test:
                return 'Lasso Regression', self.lasso

            elif maxi == self.dt_score_test:
                return 'Decision Tree', self.dt

            elif maxi == self.rf_score_test:
                return 'Random Forest', self.rf

            else:
                return 'Gradient Boosting', self.gb

        except Exception as e:
            self.logger_object.log(self.logfile,
                                   'Exception occurred in get_best_model method of the Model_Finder class.Exception '
                                   'message:  ' + str(e))
            self.logger_object.log(self.logfile,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            self.logfile.close()
            raise Exception()
