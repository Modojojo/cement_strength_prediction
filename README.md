# Concrete Strength Predicton
Predict the Compressive Strength of Concrete using Machine Learning. Built an End to End Pipeline to Fetch, Validate, Preprocess, Cluster, and then Train best performing Machine Learning Model for each of the Cluster. Best Performing Model is selected by Training 5 different Regression Algorithms on unique sets of Hyperparameters using Grid Search CV and then train the same 5 models on the Hyperparameters that gave the highest R-Squared Score, then out of these 5 Different ML Algorithms, Select the one with the Highest R-Squared Score.

## Workflow

Built an End to End Machine learning Pipeline for:
1. Fetching Raw Training Data Files from Cloud (AWS S3)
2. Validate the Files based on Certain Criteria.
3. Inserts the data from Validated Files into MongoDB Atlas database.
4. Fetches the data from MongoDB Atlas Database as one Dataframe.
5. Performs Preprocessing on the data.
6. Performs Clustering on the Data using K-Means++ and save the model to Cloud.
7. For Each Cluster Peroforms the Below Actions
    * Select 5 different Regression Algorithms
    * Trains each of the Algorithm on various Hyperparameters using Grid Search CV 
    * Finds the Best Possible Hyperparameters for each of the 5 Different Algorithms.
    * For the Obtained Hyperparameters Trains these 5 Algrithms respectively.
    * Out of these 5 Algorithms, Selects the Best Performing Algorithm using R-Squared Score, The one with Highest R-Squared score will be Selected.
    * Stores the Trained Machine Learning Model and corresponding Evaluation metrics such as Mean Squared Error, Mean Absolute Error to Cloud.
    * Also Stores a Json file that contains Cluster number to Model Mapping such that the same can be used for Prediction.

## Frontend
Created a Web App which contains functionality to make prediction on one record by filling in the feature values, start batch training/prediction process, view the Statistic for the Models trained in the previous training process and View the logs, Logs can be viewed while the process is running in the backend.

##  Project Structure 
```
main
└─── .github
|     └─── workflows
|     └─── ci-cd.yaml
|
└─── notebooks
|
└─── src
|     |──  __init__.py
|     |── cloud_connect.py
|     |── cluster_builder.py
|     |── custom_logger.py
|     |── db_connect.py
|     |── model_builder.py
|     |── prediction_validator.py
|     |── predictior.py
|     |── prepare_prediction_data.py
|     |── prepare_training_data.py
|     |── training.py
|     |── training_preprocessor.py
|     |── training_raw_validator.py
|
└─── webapp
|     |
|     └─── static
|     |     └─── css 
|     |     └─── script
|     |
|     └─── templates
|           |── dashboard.html
|           |── error_page.html
|           |── index.html
|           |── logs.html
|           |── metrics.html
|           |── prediction_completed.html
|           |── prediction_page.html
|           |── training_completed.html
|
|── Procfile
|── README.md
|── app.py
|── params.yaml
|── requirements.txt
|── setup.py
|── template.py
```
