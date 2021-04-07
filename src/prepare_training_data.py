from training_raw_validator import Validator
from db_connect import DbConnector
from cloud_connect import Cloud
import yaml


class PrepareTrainingData:
    def __init__(self, config, cloud_object, db_object):
        self.cloud = cloud_object
        self.config = config
        self.db = db_object
        self.validator = Validator()

    def read_filenames(self):
        filenames = self.cloud.get_file_names()
        return filenames

    def read_one_file(self, filename):
        file = self.cloud.read_data(filename)

        # format columns:
        new_columns = self.format_columns(file.columns)
        file.columns = new_columns
        return file

    def insert_into_db(self, file):
        self.db.insert_training_data(file)

    def validate_one_file(self, filename):
        failed = (False, None)
        if Validator.validate_file_name(filename) is True:
            file = self.read_one_file(filename)
            if Validator.validate_number_of_columns(file) is True:
                if Validator.validate_name_of_columns(file) is True:
                    try:
                        features = file.drop(self.config['base']['target_col'], axis=1)
                        label = file[self.config['base']['target_col']]
                        features = features.astype(float)
                        features.insert(len(features.columns),
                                        self.config['base']['target_col'],
                                        label)
                        success = (True, file)
                        return success
                    except Exception:
                        return failed
                else:
                    return failed
            else:
                return failed
        else:
            return failed

    def format_columns(self, columns):
        new_cols = []
        for column in columns:
            new = str(column).replace(" _", "_").lower()
            new = str(new).replace(" ", "_")
            new_cols.append(new)
        return new_cols

    def prepare_data(self):
        try:
            filenames_list = self.read_filenames()
            print(filenames_list)
            for filename in filenames_list:
                (status, file) = self.validate_one_file(filename)
                if status is True:
                    self.insert_into_db(file)
            return True
        except Exception as e:
            print(e)
            return False


if __name__ == "__main__":
    with open('params.yaml') as f:
        config = yaml.safe_load(f)
    db = DbConnector(config)
    cloud = Cloud(config)
    prep = PrepareTrainingData(config, cloud, db)
    prep.prepare_data()
