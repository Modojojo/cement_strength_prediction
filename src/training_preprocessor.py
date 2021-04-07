import numpy as np


class Preprocessor:
    def __init__(self, config):
        self.config = config

    def separate_feature_and_label(self, data):
        features = data.drop(self.config["base"]["target_col"], axis=1)
        labels = data[self.config["base"]["target_col"]]
        return features, labels

    def log_transformation(self, features):
        # some of the features have 0 as values and log(0) is not defined
        # that is why adding 1 to each value and then applying log transformation
        features = features + 1
        features = np.log(features)
        return features

    def preprocess(self, data):
        features, labels = self.separate_feature_and_label(data)
        # features = self.remove_cols_with_null_values(features)
        features = self.log_transformation(features)
        return features, labels
