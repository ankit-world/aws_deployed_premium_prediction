
# **Insurance Premium Prediction**

## Problem Statement

The goal of this project is to give people an estimate of how much they need based on their individual health situation. After that, customers can work with any health insurance carrier and its plans and perks while keeping the projected cost from our study in mind. This can assist a person in concentrating on the health side of an insurance policy rather than the ineffective part.

## Data Analysis

In the Train dataset we are provided with 23 columns(Features) of data.

Age   : age of primary beneficiary

sex  : gender of the beneficiary. It has two categories:

o Male

o Female

BMI : Body Mass Index, providing an understanding of body weights that are relatively high or low relative to height, objective index of body weight (kg/m^2) using the ratio of height to weight, ideally 18.5 to 24.9

children  : Number of children covered by the health insurance / Number of dependents.

smoker  : describing whether a person is a smoker or a non-smoker. It has 2 values: 

o Yes 

o No 

region : the beneficiary’s residential area in the US. It has 4 region values:

o Northeast 

o Southeast 

o Southwest 

o Northwest

Expenses – Individual insurance premiums billed by health insurance. 

# Approach

The main goal is to predict the health insurance premium of the user based on different factors available in the dataset.

* Data Exploration : Exploring dataset using pandas,numpy,matplotlib and seaborn.
* Data visualization : Ploted graphs to get insights about dependend and independed variables.
* Feature Engineering : Removed missing values if available, encode the categorical values using Label Encoding, scaling down the numerical values using standard scaler.
* Feature Selection : Removing not so important features by performing Back Ekimination method, VIF etc.
* Model Selection I : Tested all base models to check the base accuracy. Also ploted and calculate Performance Metrics to check whether a model is a good fit or not.
* Model Selection II : Performed Hyperparameter tuning using RandomsearchCV.
* Pickle File : Selected model as per best accuracy and created pickle file using pickle library.
* Webpage & deployment : Created a webform that takes all the necessary inputs from user and shows output. After that I have deployed project on AWS .

# Technologies Used

* Pycharm Is Used For IDE.
* For Visualization Of The Plots Matplotlib , Seaborn Are Used.
* AWS is Used For Model Deployment.
* Cassandra Database Is Used To As Data Base.
* Front End Deployment Is Done Using HTML , CSS.
* Flask is for creating the application server and pages.
* Git Hub Is Used As A Version Control System.
* json is for data validation processes.
* os is used for creating and deleting folders.
* csv is used for creating .csv format file.
* numpy is for arrays computations and mathematical operations
* pandas is for Manipulation and wrangling structured data
* scikit-learn is used for machine learning tool kit
* pickle is used for saving model
* Lasso Regression, Decision Trees, Random Forest, Gradient Boosting were used as the regression algoithms for model building


## **User Interface**

## Home Page

![image](https://user-images.githubusercontent.com/88729680/139525554-931bd2ce-5e22-4dd1-abcb-731481c69cbb.png)
![image](https://user-images.githubusercontent.com/88729680/139525561-2b5ad79b-83b1-4cec-b9aa-94678cef85e1.png)

## Predict Page

![image](https://user-images.githubusercontent.com/88729680/139525568-c28d556a-0749-4007-8876-58de0e38a48e.png)

## Insurance Premium Prediction Project Video

https://user-images.githubusercontent.com/88729680/139592015-f5989bca-7d5a-411b-9d69-953d1286e37c.mp4






## Deployments

AWS Cloud

```bash
http://awsdeployedpremiumprediction-env.eba-8h3eeubu.ap-south-1.elasticbeanstalk.com/
```

Heroku Cloud

```bash
https://ipinusrance.herokuapp.com/
```





## Run Locally

Clone the project

```bash
git clone https://github.com/ankit-world/aws_deployed_premium_prediction
```

Go to the project directory

```bash
cd aws_deployed_premium_prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```


Start the server

```bash
python app.py
```


## Usage

In Development If You Want to contribute? Great!

To fix a bug or enhance an existing module, follow these steps:

* Fork the repo

* Create a new branch
```javascript
git checkout -b new-branch
```

* Make the appropriate changes in the file

* Commit your changes
```javascript
git commit -am "New feature added"
```

* Push to the branch
```javascript
git push origin new-branch
```

* Create a pull request
```javascript
git pull
```



## Directory Structure

	└── Insurance Premium Prediction
		├── application_logging
		│   └── logging.py        
		├── Controllers
		│   └── DBconnection_info.yaml
		├── Data_Information 
		│   └── Null_Values.csv
		├── data_ingestion
		│   └── data_loader.py
		├── data_preprocessing
		│   └── preprocessing.py
		├── EDA
		│   └── EDA.ipynb
		├── model 
		│   └── Gradient Boosting
		│       └── Gradient Boosting.sav
		├── model_finder
		│   └── Model.py
		├── Prediction_logs 
		│   └── predictionlog.txt 
		├── Prediction_Output_File 
		│   └── Predictions.csv 
		├── ReplaceMissingwithNull
		│   ├── __init__.py
		│   └── transformer.py
		├── Save_Models
		│   └── save_methods.py
		├── static 
		│   ├── CSS
		│        ├── insurance.jpg
		│        ├── materialize.css
		│        └── materialize.min.css
		│   ├── js
		│        ├── materialize.js
		│        └── materialize.min.js
		├── Templates
		│   ├── index.html 
		│   └── predict.html 
		├── Tools
		│   ├── __init__.py
		│   ├── DBconnector.py
		│   ├── training_logFilescreater.py
		│   └── YamlParser.py
		├── Traning_batch_Files
		│   └── Insurance_10062021_120000.csv
		├── Training_Database_operations 
		│   ├── __init__.py
		│   └── Database_handler.py
		├── Traininig_dataValidation
		│   └── RawtrainingValidation.py 
		├── Training_FilesfromDB
		│   └── InputFile.csv 
		├── Training_Logs
		│   ├── columnValidationLog.txt
		│   ├── DataBaseConnectionLog.txt
		│   ├── DataImportExport.txt
		│   ├── Datapreprocessing_logs.txt
		│   ├── GeneralLog.txt
		│   ├── missingValuesInColumn.txt
		│   ├── ModelTrainingLog.txt
		│   ├── nameValidationLog.txt
		│   ├── TrainingDatabseInfo.txt
		│   ├── TrainingMainLog.txt
		│   ├── valuesfromSchemaValidationLog.txt
		│   └── yaml_parser.txt
		├── Training_Raw_Files_Validated
		│   ├── Good_Raw
		│           └── Insurance_10062021_120000.csv
		├── TrainingArchiveBadData
		├── app.py
		├── Directory_structure.txt
		├── prediction.py
		├── requirements.txt
		├── runtime.txt
		├── secure-connect-ankit.zip
		├── training_file.py
		├── Training_Schema.json
		└── training_val_linkage.py



## Conclusions

Here we have designed a Health Insurance Premium Prediction web application which will help people to have an idea on their health insurance needed before reaching out to any vendor for purchasing it.

Once we got the data different preprocessing techniques were performed like missing values imputations, converting categorical values to numerical values, scaling of the data, feature selection to make the data best fit for model building an prediction.

Four regression models are evaluated for individual health insurance data. The health insurance data was used to develop the three regression models, and the predicted premiums from these models were compared with actual premiums to compare the accuracies of these models. It has been found that Gradient Boosting Regression model which is built upon decision tree is the best performing model.

Various factors were used and their effect on predicted amount was examined. It was observed that a persons age and smoking status affects the prediction most in every algorithm applied. Attributes which had no effect on the prediction were removed from the features.The effect of various independent variables on the premium amount was also checked. The attributes also in combination were checked for better accuracy results.

## Documentation

[High Level Design]((https://github.com/ankit-world/aws_deployed_premium_prediction/files/7449698/High.Level.Design.pdf))

[Low Level Design]((https://github.com/ankit-world/aws_deployed_premium_prediction/files/7449699/Low.Level.Design.pdf))

[WireFrame]((https://github.com/ankit-world/aws_deployed_premium_prediction/files/7449700/WireFrame.pdf))

[DetailProject]((https://github.com/ankit-world/aws_deployed_premium_prediction/files/7449697/Detailed.Project.Report.pdf))

[Architecture]((https://github.com/ankit-world/aws_deployed_premium_prediction/files/7449696/Architecture.pdf))

## Tech Stack

**Client:** HTML, CSS

**Server:** Python 3.7


## Feedback

If you have any feedback, please reach out to me with your comments. 


## 🚀 About Me
* Data Science Enthusiastic
* AI Explorer.
