# Concrete Strength Predicton
Predict the Compressive Strength of Concrete using Machine Learning. Built an End to End Pipeline to Fetch, Validate, Preprocess, Cluster, and then Train best performing Machine Learning Model for each of the Cluster. Best Performing Model is selected by Training 5 different Regression Algorithms on unique sets of Hyperparameters using Grid Search CV and then train the same 5 models on the Hyperparameters that gave the highest R-Squared Score, then out of these 5 Different ML Algorithms, Select the one with the Highest R-Squared Score.

# Concrete Strength Predicton
Predict the Compressive Strenght of Concrete using Regression.

Built an End to End Machine learning Pipeline for:
1. Fetching Raw Training Data Files from Cloud (AWS S3)
2. Validate the Files based on Certain Criteria.
3. Inserts the data from Validated Files into MongoDB Atlas database.
4. Fetches the data from MongoDB Atlas Database as one Dataframe.
5. Performs Preprocessing on the data.
6. Performs Clustering on the Data using K-Means++
7. For Each Cluster Peroforms the Below Actions
    * Select 5 different Regression Algorithms
    * Trains each of the Algorithm on various Hyperparameters using Grid Search CV 
    * Finds the Best Possible Hyperparameters for each of the 5 Different Algorithms.
    * For the Obtained Hyperparameters Trains these 5 Algrithms respectively.
    * Out of these 5 Algorithms, Selects the Best Performing Algorithm using R-Squared Score, The one with Highest R-Squared score will be Selected.
    * Stores the Trained Machine Learning Model and corresponding Evaluation metrics such as Mean Squared Error, Mean Absolute Error to Cloud.
    * Also Stores a Json file that contains Cluster number to Model Mapping such that the same can be used for Prediction.
