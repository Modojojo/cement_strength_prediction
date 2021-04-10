from src.training_raw_validator import Validator


class PrepareTrainingData:
    def __init__(self, config, cloud_object, db_object, logger_object):
        self.logger = logger_object
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
                        success = (True, features)
                        return success
                    except Exception:
                        self.logger.log_file_validation(f"REJECTED: {filename} : Not able to convert data to float")
                        return failed
                else:
                    self.logger.log_file_validation(f"REJECTED: {filename} : Invalid Name of Columns")
                    return failed
            else:
                self.logger.log_file_validation(f"REJECTED: {filename} : Invalid Number Of Columns")
                return failed
        else:
            self.logger.log_file_validation(f"REJECTED: {filename} : Invalid File Name")
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
            self.logger.log_training_pipeline("DATA PREPARATION: Started")
            filenames_list = self.read_filenames()
            num_accepted = 0
            num_rejected = 0
            self.logger.log_training_pipeline(f"DATA PREPARATION: Fetched {len(filenames_list)} files from Server")
            for filename in filenames_list:
                (status, file) = self.validate_one_file(filename)
                if status is True:
                    self.insert_into_db(file)
                    num_accepted += 1
                else:
                    num_rejected += 1
            self.logger.log_training_pipeline(
                f"DATA PREPARATION: Completed:"
                f" Number of files Accepted = {num_accepted} :: Number of files Rejected = {num_rejected}")
            return True
        except Exception as e:
            print(e)
            return False
