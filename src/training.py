from src.cluster_builder import Cluster
from src.training_preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from src.model_builder import Model


class Training:
    def __init__(self, config, cloud, db, logger):
        self.config = config
        self.cloud = cloud
        self.db = db
        self.logger = logger

    def start_training(self):
        print("started training")
        self.logger.log_training_pipeline("TRAINING PROCESS: STARTED")
        print("logger working")
        # Fetch data from db
        self.logger.log_training_pipeline("TRAINING PROCESS: Fetching data from database")
        raw_data = self.fetch_training_data()
        self.logger.log_training_pipeline("TRAINING PROCESS: Fetched data from database")

        # preprocessing
        self.logger.log_training_pipeline("TRAINING PROCESS: Preprocessing Data")
        features, labels = self.preprocess_training_data(raw_data)
        self.logger.log_training_pipeline("TRAINING PROCESS: Preprocessing Completed")

        # Clustering
        self.logger.log_training_pipeline("TRAINING PROCESS: Creating Clusters")
        clustering_obj = Cluster(cloud_object=self.cloud, logger=self.logger)
        cluster_labels = clustering_obj.create_cluster(features=features)
        self.logger.log_training_pipeline("TRAINING PROCESS: Creating Completed")

        # combine data
        training_data = features
        training_data['cluster'] = cluster_labels
        training_data[self.config['base']['target_col']] = labels

        prediction_schema_dict = {}
        # perform Model Training based on clusters:
        for cluster_number in training_data["cluster"].unique().tolist():
            self.logger.log_training_pipeline("TRAINING PROCESS: Started Regression Model Training")
            self.logger.log_training_pipeline(f"TRAINING PROCESS: Training models for cluster {cluster_number}")
            # fetch data based on cluster number and divide into training and testing sets
            data = training_data[training_data["cluster"] == cluster_number].drop(["cluster"], axis=1)
            training_features = data.drop(self.config["base"]["target_col"], axis=1)
            training_labels = data[self.config["base"]["target_col"]]

            self.logger.log_training_pipeline("TRAINING PROCESS: Splitting data into training and development sets")
            x_train, x_test, y_train, y_test = train_test_split(training_features,
                                                                training_labels,
                                                                test_size=self.config["training_schema"]["test_size"])

            # CREATE MODEL_BUILDER OBJECT, TRAIN MODELS AND OBTAIN THE BEST MODEL
            self.logger.log_training_pipeline("TRAINING PROCESS: Entering the Model Builder Class")
            model = Model(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test, logger_object=self.logger)
            (best_model, best_model_name, best_model_metrics) = model.get_best_model()
            self.logger.log_training_pipeline(f"TRAINING PROCESS: Model training completed for cluster {cluster_number}")

            # Create model filepath for cloud storage
            model_filename = str(cluster_number) + '_' + str(best_model_name) + '/' + 'model.pkl'
            self.logger.log_training_pipeline("TRAINING PROCESS: Saving model to Cloud")
            self.cloud.save_model(best_model, model_filename)  # Save model to cloud
            self.logger.log_training_pipeline("TRAINING PROCESS: Saved model to Cloud")

            self.logger.log_training_pipeline("TRAINING PROCESS: Saving Metrics to cloud")
            self.db.save_metrics(best_model_metrics)  # Save trained model metrics in database
            self.logger.log_training_pipeline("TRAINING PROCESS: Saved Metrics to cloud")

            prediction_schema_dict[
                str(cluster_number)] = model_filename  # saving the model related to current cluster no
        self.logger.log_training_pipeline("TRAINING PROCESS: Saving Schema for prediction")
        self.cloud.write_json(prediction_schema_dict, "prediction_schema.json")  # writing prediction schema file to cloud
        self.logger.log_training_pipeline("TRAINING PROCESS: Saved Schema for prediction")
        self.logger.log_training_pipeline("TRAINING PROCESS: COMPLETED")

    def fetch_training_data(self):
        data = self.db.fetch_training_data()
        return data

    def preprocess_training_data(self, data):
        preprocessor_obj = Preprocessor(self.config)
        features, labels = preprocessor_obj.preprocess(data)
        return features, labels
