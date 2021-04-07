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
        # Fetch data from db
        raw_data = self.fetch_training_data()

        # preprocessing
        print("preprocessing")
        features, labels = self.preprocess_training_data(raw_data)

        print(features, labels)
        # Clustering
        print('clustering')
        clustering_obj = Cluster(cloud_object=self.cloud)
        cluster_labels = clustering_obj.create_cluster(features=features)
        print("clustering completed")

        # combine data
        training_data = features
        training_data['cluster'] = cluster_labels
        training_data[self.config['base']['target_col']] = labels

        print("data combined")
        prediction_schema_dict = {}
        # perform Model Training based on clusters:
        for cluster_number in training_data["cluster"].unique().tolist():
            print("training")

            # fetch data based on cluster number and divide into training and testing sets
            data = training_data[training_data["cluster"] == cluster_number].drop(["cluster"], axis=1)
            training_features = data.drop(self.config["base"]["target_col"], axis=1)
            training_labels = data[self.config["base"]["target_col"]]
            x_train, x_test, y_train, y_test = train_test_split(training_features,
                                                                training_labels,
                                                                test_size=self.config["training_schema"]["test_size"])

            # CREATE MODEL_BUILDER OBJECT, TRAIN MODELS AND OBTAIN THE BEST MODEL
            model = Model(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test, logger_object=self.logger)
            (best_model, best_model_name, best_model_metrics) = model.get_best_model()

            # Create model filepath for cloud storage
            model_filename = str(cluster_number) + '_' + str(best_model_name) + '/' + 'model.pkl'
            self.cloud.save_model(best_model, model_filename)  # Save model to cloud

            self.db.save_metrics(best_model_metrics)  # Save trained model metrics in database
            prediction_schema_dict[
                str(cluster_number)] = model_filename  # saving the model related to current cluster no

        self.cloud.write_json(prediction_schema_dict, "prediction_schema.json")  # writing prediction schema file to cloud

    def fetch_training_data(self):
        data = self.db.fetch_training_data()
        return data

    def preprocess_training_data(self, data):
        preprocessor_obj = Preprocessor(self.config)
        features, labels = preprocessor_obj.preprocess(data)
        return features, labels
