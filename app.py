from flask import Flask
import yaml
from src.db_connect import DbConnector
from src.cloud_connect import Cloud
from src.custom_logger import Logger
from src.prepare_training_data import PrepareTrainingData
from src.training import Training


app = Flask(__name__)


def read_params(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


@app.route('/', methods=["GET"])
def index():
    return


@app.route('/train', methods=["GET"])
def train():
    # create instances of cloud, database, logger
    cloud = Cloud(config)
    db = DbConnector(config)
    logger = Logger()

    # Delete previous training data from DB
    db.clear_training_folder()

    # Prepare validate and insert training raw data into DB
    data_preparation_obj = PrepareTrainingData(config=config,
                                               cloud_object=cloud,
                                               db_object=db)
    prepared = data_preparation_obj.prepare_data()

    # Fetch data from db
    # preprocess
    # Cluster
    # train Regressor models
    if prepared is True:
        trainer_obj = Training(config=config,
                               cloud=cloud,
                               db=db,
                               logger=logger)
        try:
            trainer_obj.start_training()
        except Exception as e:
            return
    return


if __name__ == "__main__":
    # fetch config
    params_path = 'params.yaml'
    global config
    config = read_params(params_path)

    # run app
    app.run()
