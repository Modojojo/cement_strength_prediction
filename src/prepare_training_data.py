from src.training_raw_validator import Validator


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
        return file

    def insert_into_db(self, file):
        self.db.insert_training_data(file)

    def validate_one_file(self, filename):
        status = True
        file = None
        return_var = (status, file)
        return return_var

    def prepare_data(self):
        filenames_list = self.read_filenames()
        for filename in filenames_list:
            (status, file) = self.validate_one_file(filename)
            if status is True:
                self.insert_into_db(file)
