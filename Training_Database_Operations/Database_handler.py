# this file helps to create the Good_training_tables

from cassandra.query import dict_factory
from application_logging.logging import logger_app
from Tools.DBconnector import DBconnector
from Tools.YamlParser import YamlParser
import csv
import os
import pandas as pd
from pathlib import Path


class Database_Operations:
    def __init__(self):
        try:
            self.logger = logger_app()
            self.f_ = open(os.path.join("Training_logs", "TrainingDatabseInfo.txt"), 'a')
            # self.goodfilepath = "Training_Raw_files_validated/Good_Raw"
            self.goodfilepath = os.path.join("Training_Raw_files_validated", "Good_Raw")
            # self.yaml_path = "Controllers/DBconnection_info.yaml"
            self.yaml_path = os.path.join("Controllers", "DBconnection_info.yaml")
            self.session = DBconnector().connect()
            self.key_space = YamlParser(self.yaml_path).yaml_parser()[0]['Good_training_tables_info']['keyspace_name']
            self.Good_training_TableName = YamlParser(self.yaml_path).yaml_parser()[0]['Good_training_tables_info'][
                'table_name']
            self.logger.log(self.f_, 'All the required Files are initialized ....')
            self.f_.close()
        except Exception as e:
            self.logger.log(file_object=self.f_, log_massage="Exception : " + str(e))
            self.f_.close()
            raise e

    def CreateGoodTraining_table(self):
        """ Task : this method will create the GoodTraining Table to store the Validated data.
            Inputs : Receives the Inputs from "DBconnection_info.yaml" files
            Ip/1 : database connector pointer
            Ip/2 : Good-Training-TableName
            Ip/3 : particular keyspace
            return : None
        """
        try:
            self.f_ = open(os.path.join("Training_logs", "TrainingDatabseInfo.txt"), 'a')
            self.logger.log(self.f_, 'Entered into the class : CreateGoodTraining_table ....')
            self.session.execute(f"use {self.key_space}")
            self.session.execute(
                f"CREATE TABLE if not exists {self.Good_training_TableName} (age  int , sex  text , bmi  float , "
                f"children  int , smoker  text , region  text , charges  float, PRIMARY KEY(age,sex,bmi,children,"
                f"smoker,region,charges)) ;")
            self.logger.log(self.f_, f"{self.Good_training_TableName}  created Successfully !....")
            self.f_.close()
        except Exception as e:
            self.logger.log(file_object=self.f_, log_massage="Exception : " + str(e))
            self.f_.close()
            raise e

    def DataInsertion(self):
        """
                Method Name: DataInsertion
                Description: This method inserts the Good data files from the Good_Raw folder into the
                            above created table.
                Output: None
                On Failure: Raise Exception


        """

        goodfilepath = self.goodfilepath
        log_f = open(os.path.join("Training_Logs", "DataImportExport.txt"), 'a')
        self.logger.log(log_f, 'Entered Into Class : DataBaseOperations > DataInsertion ')
        self.logger.log(log_f, f'Files found in Good Raw dirs  :{str(os.listdir(self.goodfilepath))}')

        try:
            self.session.execute(f"use {self.key_space}")
            files = [file_ for file_ in os.listdir(self.goodfilepath)]
            for file_ in files:
                f_path = os.path.join(goodfilepath, file_)
                # pandas operations
                data = pd.read_csv(f_path)
                a = data.values
                try:
                    for record in a:
                        l = list(record)
                        Query_2 = f"Insert into {self.Good_training_TableName} (age,sex,bmi,children,smoker,region," \
                                  f"charges) values ({l[0]} , '{l[1]}' ,  {l[2]} ,  {l[3]} , '{l[4]}'  ,  '{l[5]}' ,  " \
                                  f"{l[6]}); "
                        self.session.execute(Query_2)

                    self.logger.log(log_f, log_massage="Data Uploaded successfully ....")
                    log_f.close()

                except Exception as e:
                    self.logger.log(log_f, 'Exception Occurs in Data Insertion Process  ... . ')
                    self.logger.log(log_f, 'Exception ' + str(e))
                    raise e

        except Exception as e:
            self.logger.log(file_object=log_f, log_massage='Exception : ' + str(e))
            log_f.close()
            raise e

    def DataImport(self):
        """
                MethodName : DataImport
                Task : This  method will Import the Data from  Table & convert / store this data into the Csv Format .
                Output file location : 'Training_FileFromDB/'
                return : None


        """
        self.fileFromDb = 'Training_FileFromDB/'
        self.fileName = 'InputFile.csv'  # the file In which data will be storing
        log_f = open(os.path.join("Training_Logs", "DataImportExport.txt"), 'a')  # Log File to Capture Process
        self.logger.log(log_f, 'Entered Into Class : DataBaseOperations > DataImport ')

        # creating the Dir to store the Data
        if not os.path.isdir(self.fileFromDb):
            os.makedirs(self.fileFromDb)

        try:
            self.session.execute(f"use {self.key_space}")
            Query = f"select * from {self.Good_training_TableName} ;"
            self.session.row_factory = dict_factory
            rows = list(self.session.execute(Query))
            # storing the data into csv file
            try:
                fieldnames = ['age', 'sex', 'bmi', 'children', 'smoker',
                              'region', 'charges'
                              ]
                with open(self.fileFromDb + self.fileName, mode='w', newline="") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                self.logger.log(file_object=log_f,
                                log_massage=f" Data Imported in to file {os.path.join(self.fileFromDb, self.fileName)}")
                log_f.close()

            except Exception as e:
                self.logger.log(log_f, log_massage="Exception : Method : DataImport => " + str(e))
                log_f.close()
                raise e

        except Exception as e:
            self.logger.log(log_f, log_massage="Exception : Method : DataImport => " + str(e))
            log_f.close()
            raise e
