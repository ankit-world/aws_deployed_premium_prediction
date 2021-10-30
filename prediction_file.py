# this module will predict the o./p
from data_ingestion import data_loader
from Save_Model import save_methods
from application_logging.logging import logger_app
import pandas as pd
from data_preprocessing import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")


class prediction:
    def __init__(self):
        self.log_writer = logger_app()
        self.filepath = os.path.join("Prediction_Logs", "PredictionLog.txt")

    def convert_input_into_data(self, user_input):
        """
            Method : convert_input_into_data
            Aim : "This method connects modules [Preprocessing + Model_Finder + Save_Model]"
                    1. Preprocess the data
                        1.1 Perform null values imputing
                        1.2 Encoding of Categorical values
                        1.3 Scaling the input data
                        1.4 Removing the unnecessary columns

            Return : None
            On fault : raise the Exception .


        """
        self.file_object = open(self.filepath, 'a')
        self.log_writer.log(self.file_object, 'Enter In Convert_input_into_data')
        self.input = user_input
        # print(self.input)
        try:
            preprocessor = preprocessing.Preprocessor()
            df = pd.DataFrame(self.input, index=["age", "sex", "bmi", "children", "smoker", "region"])
            data = df.transpose()
            data['age'] = data['age'].astype('int64')
            data['bmi'] = data['bmi'].astype('float64')
            data['children'] = data['children'].astype('int64')
            # print(data.info())
            # print(data)
            """
            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation
            """
            # Convert categorical values to numeric values
            data = preprocessor.encode_categorical_columns(data)

            """
            #Concatenating categorical values
            X = pd.concat([scaled_num_df, cat_df], axis=1)
            print(X)
            """

            # As we are accepting a single record , hence we  need to fit the scaler using available training data on
            # the new data for which we need to predict the output
            data_getter = data_loader.Data_Getter()
            data_sc = data_getter.get_data()
            data_sc = preprocessor.encode_categorical_columns(data_sc)
            data_sc.drop('charges', axis=1, inplace=True)
            sc = StandardScaler()
            sc.fit(data_sc)
            pred_data = sc.transform([list(data.iloc[0])])
            pred_data_scale = pd.DataFrame(pred_data, columns=data.columns)
            # print(pred_data_scale)

            # remove the unnamed column as it doesn't contribute to prediction.
            X = preprocessor.remove_columns(pred_data_scale, ["sex"])

            self.file_object.close()
            return X

        except Exception as error:
            self.log_writer.log(self.file_object,
                                'Error occurred while running the convert_input_into_data!! Error:: %s' % error)
            self.file_object.close()
            raise error

    def get_prediction(self, data):
        """
            Method : convert_input_into_data
            Aim : "This method connects modules [Save_Model]"
                    1. load the ML model on the input data
                    2. Will get the output as prediction
                    3. csv file will be created with that prediction output

            Return : None
            On fault : raise the Exception .


        """
        # Logging the start of Prediction
        self.file_object = open(self.filepath, 'a')
        self.log_writer.log(self.file_object, 'Start of get_prediction')
        self.data = data
        try:
            model_loader = save_methods.Model_Operation()
            model_name = model_loader.find_correct_model_file()
            model = model_loader.load_model(model_name)
            result = list(model.predict(self.data))
            result = pd.DataFrame(result, columns=['Prediction'])
            path = os.path.join("Prediction_Output_File", "Predictions.csv")
            result.to_csv(path, header=True, mode='a+')
            self.log_writer.log(self.file_object, 'End of get_prediction')
            self.file_object.close()

        except Exception as error:
            self.log_writer.log(self.file_object, 'Error occurred while running the prediction!! Error:: %s' % error)
            self.file_object.close()
            raise error
