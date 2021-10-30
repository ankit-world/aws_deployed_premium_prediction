# **Insurance Premium Prediction**

# **Problem Statement**

The goal of this project is to give people an estimate of how much they need based on their individual health situation. After that, customers can work with any health insurance carrier and its plans and perks while keeping the projected cost from our study in mind. This can assist a person in concentrating on the health side of an insurance policy rather than the ineffective part.

# **Data Analysis**

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

# **Approach**

The main goal is to predict the health insurance premium of the user based on different factors available in the dataset.

* Data Exploration : Exploring dataset using pandas,numpy,matplotlib and seaborn.
* Data visualization : Ploted graphs to get insights about dependend and independed variables.
* Feature Engineering : Removed missing values if available, encode the categorical values, scaling down the numerical values.
* Feature Selection : Removing not so important features by performing Back Ekimination method, VIF etc.
* Model Selection I : Tested all base models to check the base accuracy. Also ploted and calculate Performance Metrics to check whether a model is a good fit or not.
* Model Selection II : Performed Hyperparameter tuning using RandomsearchCV.
* Pickle File : Selected model as per best accuracy and created pickle file using pickle library.
* Webpage & deployment : Created a webform that takes all the necessary inputs from user and shows output. After that I have deployed project on AWS .

# **Technologies Used**

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


# **User InterFace**

* Home Page

![image](https://user-images.githubusercontent.com/88729680/139525554-931bd2ce-5e22-4dd1-abcb-731481c69cbb.png)
![image](https://user-images.githubusercontent.com/88729680/139525561-2b5ad79b-83b1-4cec-b9aa-94678cef85e1.png)

* Predict Page

![image](https://user-images.githubusercontent.com/88729680/139525568-c28d556a-0749-4007-8876-58de0e38a48e.png)


# **Insurance Premium Prediction Project Video**

https://user-images.githubusercontent.com/88729680/139559119-c22cfb09-776d-411e-a671-7e4a4173596f.mp4



