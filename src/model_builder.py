from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class Model:
    def __init__(self, train_x, train_y, test_x, test_y, logger_object):
        """
        Instantiate Model class for Obtaining the Best Possible model for the specified Dataset
        Author: Modojojo
        :param train_x: Training Features
        :param train_y: Training Labels
        :param test_x: Testing Features
        :param test_y: Testing Labels
        :param logger_object: Object of Logger Class (Custom Logging Functionality)
        """
        self.logger = logger_object
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.param_grid_xgboost = {'max_depth': range(3, 12)}
        self.param_grid_random_forest = {'n_estimators': range(50, 150, 10),
                                         'criterion': ['mse', 'mae'],
                                         'max_depth': range(3, 8)}
        self.param_grid_linear_regression = {'normalize': [True, False]}
        self.param_grid_svr = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}
        self.param_grid_decision_tree = {'criterion': ['mse', 'mae'],
                                         'max_depth': range(3, 8),
                                         'splitter': ['random', 'best']}

    def train_xgboost(self):
        """
        Trains XGBoost Regressor with the best possible parameters obtained from GRID SEARCH CV
        Calls: get_best_params_xgboost(), logger.log_training()
        Author: Modojojo
        :return: Model Object if Trained successfully else return False
        """
        try:
            self.logger.log_training('Finding Best Hyper-Parameters for XGBoost')
            model_params = self.get_best_params_xgboost()   # Obtain Best Obtained Parameters for XGBoost
            if model_params is not False:
                max_depth = model_params
                model = XGBRegressor(max_depth=max_depth)   # Instantiate XGBoost on obtained parameters
                self.logger.log_training(
                    'Training XGBoost Regressor with the Following HyperParameters :: '
                    'max_depth: {}'.format(max_depth)
                )
                model.fit(self.train_x, self.train_y)   # Fit data and train model
                return model
            else:
                raise Exception('FAILED: Error while training Model for XGBoost')
        except Exception:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training XGBoost Model')
            return False

    def train_random_forest(self):
        """
        Trains Random Forest Regressor with the best possible parameters obtained from GRID SEARCH CV
        Calls: get_best_params_random_forest(), logger.log_training()
        Author: Modojojo
        :return: Model Object if Trained successfully else return False
        """
        try:
            self.logger.log_training('Finding Best Hyper-Parameters for Random Forest')
            model_params = self.get_best_params_random_forest()     # Obtain Best parameters
            if model_params is not False:
                (max_depth, n_estimators, criterion) = model_params
                model = RandomForestRegressor(max_depth=max_depth,  # Instantiate Model
                                              n_estimators=n_estimators,
                                              criterion=criterion)
                self.logger.log_training(
                    'Training Random Forest Regressor with the Following HyperParameters ::'
                    ' max_depth: {}, n_estimators: {}, criterion: {}'.format(max_depth, n_estimators, criterion)
                )
                model.fit(self.train_x, self.train_y)   # Fit data and Train Model
                return model
            else:
                raise Exception('Error while training the best Model for Random Forest')
        except Exception:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training Random Forest Model')
            return False

    def train_linear_regression(self):
        """
        Trains Linear Regression Model with the best possible parameters obtained from GRID SEARCH CV
        Calls: get_best_params_linear_regression(), logger.log_training()
        Author: Modojojo
        :return: Model Object if Trained successfully else return False
        """
        try:
            self.logger.log_training('Finding Best Hyper-Parameters for Linear Regression')
            model_params = self.get_best_params_linear_regression()     # Obtain Best parameters
            if model_params is not False:
                normalize = model_params
                model = LinearRegression(normalize=normalize)   # Instantiate Model
                self.logger.log_training(
                    'Training Linear Regression with the Following HyperParameters :: Normalize: {}'.format(
                        normalize
                    )
                )
                model.fit(self.train_x, self.train_y)   # Fit data and Train Model
                return model
            else:
                raise Exception('Error while training the best Model for Linear Regression')
        except Exception:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training Linear Regression Model')
            return False

    def train_decision_tree(self):
        """
        Trains Decision Tree Regressor with the best possible parameters obtained from GRID SEARCH CV
        Calls: get_best_param_decision_tree(), logger.log_training()
        Author: Modojojo
        :return: Model Object if Trained successfully else return False
        """
        try:
            self.logger.log_training('Finding best Hyper-Parameters for Decision Tree')
            model_params = self.get_best_param_decision_tree()      # Obtain Best parameters
            if model_params is not False:
                (max_depth, criterion, splitter) = model_params
                model = DecisionTreeRegressor(max_depth=max_depth,  # Instantiate Model
                                              criterion=criterion,
                                              splitter=splitter)
                self.logger.log_training('Training Decision Tree with the Following HyperParameters ::'
                                         ' max_depth: {}, criterion: {}, splitter: {}'.format(max_depth,
                                                                                              criterion,
                                                                                              splitter)
                                         )
                model.fit(self.train_x, self.train_y)   # Fit data and Train Model
                return model
            else:
                raise Exception('Error while training the best Model for Decision Tree')
        except Exception as e:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training Decision Tree Model')
            return False

    def train_svr(self):
        """
        Trains Support Vector Machine with the best possible parameters obtained from GRID SEARCH CV
        Calls: get_best_param_svr()
        Author: Modojojo
        :return: Model Object if Trained successfully else return False
        """
        try:
            self.logger.log_training('Finding best Hyper-Parameters for Support Vector Machine')
            model_params = self.get_best_param_svr()    # Obtain Best parameters
            if model_params is not False:
                kernel = model_params
                model = SVR(kernel=kernel)  # Instantiate Model
                self.logger.log_training('Training Support Vector Regressor with the Following HyperParameters ::'
                                         ' kernel: {}'.format(kernel)
                                         )
                model.fit(self.train_x, self.train_y)   # Fit data and Train Model
                return model
            else:
                raise Exception('Error while training the best Model for Support Vector Machine')
        except Exception:
            self.logger.log_training('ERROR: MODEL SELECTION: Error while training Support Vector Machine Model')
            return False

    def get_best_params_xgboost(self):
        """
        Uses GridSearchCV to train XGBoost on unique sets of Parameters defined in initialization function
        Author: Modojojo
        :return: Best Parameters if trained successfully, else False
        """
        try:
            model = XGBRegressor(objective='reg:linear')    # Instantiate And use XGBoost model

            # Initialize GridSearchCV object with Specified model and Param Grid
            grid = GridSearchCV(model,
                                param_grid=self.param_grid_xgboost,
                                cv=5, verbose=3)
            grid.fit(self.train_x, self.train_y)    # Fit Data to Grid Search CV

            max_depth = grid.best_params_['max_depth']  # Obtain the Params for which the model performed the best

            retvar = max_depth
            return retvar
        except Exception as e:
            self.logger.log_training('MODEL SELECTION: Error while getting best Parameters for XGBoost')
            return False

    def get_best_params_random_forest(self):
        """
        Uses GridSearchCV to train Random Forest on unique sets of Parameters defined in initialization function
        Author: Modojojo
        :return: Best Parameters if trained successfully, else False
        """
        try:
            model = RandomForestRegressor()     # Instantiate Random Forest Model

            # Initialize GridSearchCV object with Specified model and Param Grid
            grid = GridSearchCV(model,
                                param_grid=self.param_grid_random_forest,
                                cv=5, verbose=3)
            grid.fit(self.train_x, self.train_y)    # Fit Data to Grid Search CV

            # Obtain the Params for which the model performed the best
            n_estimators = grid.best_params_['n_estimators']
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            retvar = (max_depth, n_estimators, criterion)
            return retvar
        except Exception:
            self.logger.log_training('MODEL SELECTION: Error while getting best Parameters for Random Forest')
            return False

    def get_best_params_linear_regression(self):
        """
        Uses GridSearchCV to train Linear Regression on unique sets of Parameters defined in initialization function
        Author: Modojojo
        :return: Best Parameters if trained successfully, else False
        """
        try:
            model = LinearRegression()  # Instantiate Linear Regression Model

            # Initialize GridSearchCV object with Specified model and Param Grid
            grid = GridSearchCV(model,
                                param_grid=self.param_grid_linear_regression,
                                cv=5, verbose=3)

            grid.fit(self.train_x, self.train_y)    # Fit Data to Grid Search CV
            normalize = grid.best_params_['normalize']  # Obtain the Params for which the model performed the best
            return normalize
        except Exception:
            self.logger.log_training(
                'MODEL SELECTION: Error while getting best Parameters for Linear Regression'
            )
            return False

    def get_best_param_svr(self):
        """
        Uses GridSearchCV to train Support Vector Machine on unique set of Parameters defined in initialization function
        Author: Modojojo
        :return: Best Parameters if trained successfully, else False
        """
        try:
            model = SVR()   # Initialize the Support Vector Machine Model

            # Initialize GridSearchCV object with Specified model and Param Grid
            grid = GridSearchCV(model,
                                param_grid=self.param_grid_svr,
                                cv=5, verbose=3)

            grid.fit(self.train_x, self.train_y)    # Fit Data to Grid Search CV

            kernel = grid.best_params_['kernel']    # Obtain the Params for which the model performed the best

            retvar = kernel
            return retvar
        except Exception as e:
            self.logger.log_training(
                'MODEL SELECTION: Error while getting best Parameters for Support Vector Machine'
            )
            return False

    def get_best_param_decision_tree(self):
        """
        Uses GridSearchCV to train Decision Tree on unique sets of Parameters defined in initialization function
        Author: Modojojo
        :return: Best Parameters if trained successfully, else False
        """
        try:
            model = DecisionTreeRegressor()       # Instantiating Decision Tree Model

            # Initialize GridSearchCV object with Specified model and Param Grid
            grid = GridSearchCV(model,
                                param_grid=self.param_grid_decision_tree,
                                cv=5, verbose=3)

            grid.fit(self.train_x, self.train_y)    # Fit Data to Grid Search CV

            # Obtain the Params for which the model performed the best
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            splitter = grid.best_params_['splitter']

            retvar = (max_depth, criterion, splitter)
            return retvar
        except Exception:
            self.logger.log_training(
                'MODEL SELECTION: Error while getting best Parameters for Decision Tree'
            )
            return False

    def get_best_model(self):
        """
        1. Trains all the Possible models defined in the class
        2. Makes prediction for test set and check performance using r-squared score
        3. Select the model with the maximum R-squared score
        Author: Modojojo
        :return: Best obtained model Object, Best Performing Model Name, Metrics
        """
        try:
            self.logger.log_training('MODEL SELECTION: Multi Model Training -- STARTED')
            # Trains all the possible models defined in above functions
            xgboost_regressor = self.train_xgboost()
            random_forest_regressor = self.train_random_forest()
            linear_regression_regressor = self.train_linear_regression()
            decision_tree_rregressor = self.train_decision_tree()
            svr = self.train_svr()
            self.logger.log_training('MODEL SELECTION: Multi Model Training -- COMPLETED')

            # Predefined dictionary for model names
            models_dictionary = {'XGBoost': xgboost_regressor,
                                 'Random_Forest': random_forest_regressor,
                                 'Linear_Regression': linear_regression_regressor,
                                 'Decision_Tree': decision_tree_rregressor,
                                 'Support_Vector_Machine': svr}

            # Dictionary for storing R-squared Scores for corresponding model
            scores_dictionary = {'XGBoost': None,
                                 'Random_Forest': None,
                                 'Logistic_Regression': None,
                                 'Decision_Tree': None,
                                 'Support_Vector Machine': None}

            self.logger.log_training('MODEL SELECTION: Making Predictions and Storing Scores -- STARTED')

            # For each Successfully trained model
            # Obtain Predictions and use those predictions to evaluate model using R-Squared score
            # Store the obtained R-squared score in scores_dictionary if the process was successful
            # if any error occurs, Doesnt store scores. Default score available in scores_dictionary will be None
            for model_name in models_dictionary:
                model = models_dictionary[model_name]
                if model is not False:  # Checking if model is trained properly or not
                    prediction = model.predict(self.test_x)     # Make Predictions using test set
                    try:
                        scores_dictionary[model_name] = r2_score(self.test_y, prediction)   # Obtain and Store R-Squared Score
                    except Exception as e:  # Logs message if any Exception/Error occurs
                        self.logger.log_training(
                            'MODEL SELECTION : FAILED to select by R-squared Score : {}'.format(str(e))
                        )

            self.logger.log_training('MODEL SELECTION: Making Predictions and Storing Scores -- COMPLETED')

            self.logger.log_training('MODEL SELECTION: Comparing and Finding Best Model')

            # Select Models and scores of models for which the scores are available
            final_scores_dict = {}
            for model_name in scores_dictionary:
                if scores_dictionary[model_name] is not None:   # if the Score obtained for the model is Not 'None'
                    final_scores_dict[model_name] = scores_dictionary[model_name]   # Store model name, r-squared score

            self.logger.log_training('Trained Following Models with given R-Squared Scores: {}'.format(final_scores_dict))

            # Obtain the model Name for the model that has the Highest R-Squared Score
            best_model_name = max(final_scores_dict, key=lambda x: final_scores_dict[x])

            # Select the Best Model
            best_model = models_dictionary[best_model_name]
            self.logger.log_training('MODEL SELECTION: COMPLETED: selected -- {}'.format(best_model_name))

            # Obtain various Evaluation Metrics for the Final Selected Model
            best_model_metrics = self.metrics(best_model, best_model_name)

            # Return MODEL, MODEL NAME, METRICS for the final selected model
            retvar = (best_model, best_model_name, best_model_metrics)
            return retvar

        except Exception as e:
            self.logger.log_training('CRITICAL: * Training Failed * please check the previous log for reason')
            raise e

    def metrics(self, best_model, best_model_name):
        """
        Evaluates the model on certain scores, store them and return them
        Author: Modojojo
        :param best_model: Object of the Model
        :param best_model_name: Name of the Model
        :return: Dictionary: contains performance metrics for inputted model
        """
        self.logger.log_training("Preparing Metrics to be stored and reviewed")
        predictions = best_model.predict(self.test_x)   # Make prediction using test set
        r2 = r2_score(self.test_y, predictions)         # Obtain R-Squared Score
        mae = mean_absolute_error(self.test_y, predictions)     # Obtain Mean Absolute Error
        mse = mean_squared_error(self.test_y, predictions)      # valuate Mean Squared Error
        metrics_dict = {'model': str(best_model_name),          # Store all the metrics in dictionary
                        'R-Squared Score': str(r2),
                        'Mean Squared Error': str(mse),
                        'Mean Absolute Error': str(mae)}
        return metrics_dict
