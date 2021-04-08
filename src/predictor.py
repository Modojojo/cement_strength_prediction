from src.cluster_builder import Cluster
import numpy as np


class Predictor:
    def __init__(self, config, cloud, db, logger):
        self.config = config
        self.cloud = cloud
        self.db = db
        self.logger = logger

    def fetch_prediction_data(self):
        data = self.db.fetch_prediction_data()
        return data

    def preprocess_data(self, features):
        features = self.log_transformation(features)
        return features

    def predict_cluster(self, features):
        cluster_obj = Cluster(self.cloud, self.logger)
        cluster_predictions = cluster_obj.predict_clusters(features)
        return cluster_predictions

    def predict_target(self, features, cluster):
        prediction_schema = self.cloud.load_json(self.config['cloud']['prediction_schema'])
        final_predictions = {}
        index = range(0, len(features))
        features['index'] = index
        features['cluster'] = cluster
        for cluster_number in features['cluster'].unique().tolist():
            x_id = features[features['cluster'] == cluster_number]['index'].tolist()
            x = features[features['cluster'] == cluster_number].drop(['index', 'cluster'], axis=1)
            model_name = prediction_schema[str(cluster_number)]
            model = self.cloud.load_model(model_name)
            predictions = model.predict(x)

            # put predictions in final_predictions dictionary
            for i in range(len(predictions)):
                final_predictions[str(x_id[i])] = predictions[i]

        return final_predictions

    def predict(self):
        print("fetching data from db")
        features = self.fetch_prediction_data()
        print("preprocessing data")
        features = self.preprocess_data(features)
        print("predicting clusters")
        clusters = self.predict_cluster(features)
        print("predicting target")
        final_predictions = self.predict_target(features, clusters)
        print("inserting into db")
        self.db.insert_predictions(final_predictions)
        print("completed")

    def predict_one(self, features):
        print("prediction for one record started")
        features = np.array(features)
        features = self.preprocess_data(features)
        cluster_id = self.predict_cluster(features)
        prediction_schema = self.cloud.load_json(self.config['cloud']['prediction_schema'])
        model_name = prediction_schema[str(cluster_id)]
        model = self.cloud.load_model(model_name)
        prediction = model.predict(features)
        print('completed')
        return prediction[0]

    def log_transformation(self, features):
        # some of the features have 0 as values and log(0) is not defined
        # that is why adding 1 to each value and then applying log transformation
        features = features + 1
        features = np.log(features)
        return features
