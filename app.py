# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import cross_origin
from training_val_linkage import Main_trainingDataValidation
import training_file as train
import prediction_file as pred
from data_ingestion import data_loader
from application_logging.logging import logger_app
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    """
    This will provide you the homepage of the Application
    """
    return render_template("index.html")


@app.route('/train', methods=['GET'])
@cross_origin()
def training():
    """
    Method : training
    Description : 1. Perform training data validation
                  2. Perform model training
    """
    file_object = open(os.path.join("Training_Logs", "TrainingMainLog.txt"), 'a+')
    log_writer = logger_app()
    try:
        train_val_obj = Main_trainingDataValidation()
        train_val_obj.train_validation()
        log_writer.log(file_object, 'Data Validation Completed Successfully')
        train_obj = train.training()
        train_obj.trainingModel()
        log_writer.log(file_object, 'Model Training Completed Successfully')
        file_object.close()
    except Exception as e:
        log_writer.log(file_object, 'This is a problem with Model Training' + str(e))
        file_object.close()
        raise Exception(str(e))
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def predict():
    """
    Method : predict Description: With this method user will be able to take values from the user and able to predict
    the output based upon it
    """
    msg = ''
    if request.method == 'POST' and 'age' in request.form and 'sex' in request.form and 'bmi' in request.form and \
            'children' in request.form and 'smoker' in request.form and 'region' in request.form:
        file_object = open("Prediction_Logs/PredictionLog.txt", 'a+')
        log_writer = logger_app()
        log_writer.log(file_object, 'Start For Gathering Data for prediction')
        try:
            # reading the inputs given by the user

            age = (request.form['age'])
            sex = (request.form['sex'])
            bmi = (request.form['bmi'])
            children = (request.form['children'])
            smoker = (request.form['smoker'])
            region = (request.form['region'])
            if not age and not sex and not bmi and not children and not smoker and not region:
                msg = 'Please fill out the form !'
                return render_template('index.html', msg=msg)
            elif not age or not sex or not bmi or not children or not smoker or not region:
                msg = 'Please fill all the required inputs!'
                return render_template('index.html', msg=msg)
            else:
                p = pred.prediction()  # Predict A File
                data = p.convert_input_into_data([int(age), sex, float(bmi), int(children), smoker, region])
                # print(data)
                p.get_prediction(data)
                log_writer.log(file_object, 'Start For Prediction')

                predict = data_loader.Data_Getter()
                prediction = predict.prediction_data()

                # The below logic will pick only the latest predicted record
                l = list(prediction['Prediction'])
                for i in l:
                    if i == 'Prediction':
                        l.remove('Prediction')
                res_pred = l[-1]
                # print(res_pred)

                # showing the prediction results in a UI
                return render_template('predict.html', prediction=res_pred)
                # return prediction
                log_writer.log(file_object, "Prediction process Completed...")
                file_object.close()

        except Exception as e:
            log_writer.log(file_object, 'This is a problem with Prediction Process' + str(e))
            file_object.close()
            raise Exception(str(e))

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
