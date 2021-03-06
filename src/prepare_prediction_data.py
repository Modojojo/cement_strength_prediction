from src.prediction_validator import Validator


class PreparePredictionData:
    def __init__(self, config, cloud_object, db_object, logger_object):
        self.cloud = cloud_object
        self.logger = logger_object
        self.config = config
        self.db = db_object
        self.validator = Validator()

    def read_filenames(self):
        filenames = self.cloud.get_file_names(prediction=True)
        return filenames

    def read_one_file(self, filename):
        file = self.cloud.read_data(filename,predicton=True)

        # format columns:
        new_columns = self.format_columns(file.columns)
        file.columns = new_columns
        return file

    def insert_into_db(self, file):
        self.db.insert_prediction_data(file)

    def validate_one_file(self, filename):
        failed = (False, None)
        if Validator.validate_file_name(filename) is True:
            file = self.read_one_file(filename)
            if Validator.validate_number_of_columns(file) is True:
                if Validator.validate_name_of_columns(file) is True:
                    try:
                        file = file.astype(float)
                        success = (True, file)
                        return success
                    except Exception:
                        self.logger.log_file_validation(f"{filename} : REJECTED: Failed to convert data types to fioat")
                        return failed
                else:
                    self.logger.log_file_validation(f"{filename} : REJECTED: Incorrect Column Names")
                    return failed
            else:
                self.logger.log_file_validation(f"{filename} : REJECTED: Incorrect Number of columns")
                return failed
        else:
            self.logger.log_file_validation(f"{filename} : REJECTED: Incorrect filename")
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
            self.logger.log_training_pipeline("PREPARING TRAINING DATA : Started : Please check file validation logs for more information")
            filenames_list = self.read_filenames()
            for filename in filenames_list:
                (status, file) = self.validate_one_file(filename)
                if status is True:
                    self.insert_into_db(file)
            self.logger.log_training_pipeline("PREPARING TRAINING DATA : Completed")
            return True
        except Exception as e:
            self.logger.log_training_pipeline("Failed to prepare Training Data")
            return False
