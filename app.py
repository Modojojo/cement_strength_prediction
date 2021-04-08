from flask import Flask, request, render_template
import yaml
import os
from src.db_connect import DbConnector
from src.cloud_connect import Cloud
from src.custom_logger import Logger
from src.prepare_training_data import PrepareTrainingData
from src.training import Training
from src.prepare_prediction_data import PreparePredictionData
from src.predictor import Predictor

params_path = 'params.yaml'
webapp_root = 'webapp'

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


def read_params(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/training_dashboard', methods=["GET"])
def dashboard():
    return render_template('dashboard.html')


@app.route('/prediction_page', methods=["GET", "POST"])
def prediction_page():
    try:
        if request.method == "GET":
            return render_template('prediction_page.html')
        elif request.method == "POST":

            cloud = Cloud(config)
            db = DbConnector(config)
            logger = Logger()

            features = [int(request.form["component1"]),
                        int(request.form["component2"]),
                        int(request.form["component3"]),
                        int(request.form["component4"]),
                        int(request.form["component5"]),
                        int(request.form["component6"]),
                        int(request.form["component7"]),
                        int(request.form["component8"])]

            predictor_obj = Predictor(config, cloud, db, logger)
            prediction_single = predictor_obj.predict_one(features)
            prediction_single = str(prediction_single)[:6] + ' ' + 'Mega Pascal'
            db.close()
            logger.close()
            return render_template('prediction_page.html', prediction=prediction_single)
    except Exception as e:
        return render_template("error_page.html", message=str(e))


@app.route('/train', methods=["GET"])
def train():
    try:
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
        print("prepared  : {}".format(prepared))
        if prepared is True:
            trainer_obj = Training(config=config,
                                   cloud=cloud,
                                   db=db,
                                   logger=logger)
            trainer_obj.start_training()
        return render_template('training_completed.html')
    except Exception as e:
        return render_template("error_page.html", message=str(e))


@app.route('/predict', methods=["GET"])
def prediction():
    try:
        cloud = Cloud(config)
        db = DbConnector(config)
        logger = Logger()

        # delete previous data
        db.clear_prediction_folder()

        # prepare data
        data_prep = PreparePredictionData(config=config,
                                          cloud_object=cloud,
                                          db_object=db)
        data_prep.prepare_data()

        predictor_obj = Predictor(config, cloud, db, logger)
        predictor_obj.predict()

        db.close()
        logger.close()

        return render_template('prediction_completed.html')
    except Exception as e:
        return render_template("error_page.html", message=str(e))


@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        db = DbConnector(config)
        metrics = db.fetch_metrics()
        db.close()
        for metric in metrics:
            for i in metric:
                if len(metric[i]) > 5 and i != 'model' and i != 'date':
                    metric[i] = metric[i][:5]

        return render_template('metrics.html', metrics=metrics)

    except Exception as e:
        print(e)
        return render_template('404.html', message=str(e))


@app.route('/logs', methods=["POST"])
def get_logs():
    return render_template("logs.html")


if __name__ == "__main__":
    # fetch config
    params_path = 'params.yaml'
    global config
    config = read_params(params_path)

    # run app
    app.run()
