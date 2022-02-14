import pandas as pd
import numpy as np
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation


class prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            #code change
            # wafer_names=data['Wafer']
            # data=data.drop(labels=['Wafer'],axis=1)

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            data = preprocessor.remove_columns(data, [
                'education'])  # remove the column as it doesn't contribute to prediction.
            data = preprocessor.remove_unwanted_spaces(data)  # remove unwanted spaces from the dataframe
            data.replace('?', np.NaN, inplace=True)  # replacing '?' with NaN values for imputation

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation

            # Proceeding with more data pre-processing steps
            scaled_num_df = preprocessor.scale_numerical_columns(data)
            cat_df = preprocessor.encode_categorical_columns(data)
            X = pd.concat([scaled_num_df, cat_df], axis=1)


            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans=file_loader.load_model('KMeans')

            ##Code changed
            #pred_data = data.drop(['Wafer'],axis=1)
            clusters=kmeans.predict(X)#drops the first column for cluster prediction
            X['clusters']=clusters
            clusters=X['clusters'].unique()
            predictions=[]
            for i in clusters:
                cluster_data= X[X['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                result=(model.predict(cluster_data))
                for res in result:
                    if res==0:
                        predictions.append('<=50K')
                    else:
                        predictions.append('>50K')

            final= pd.DataFrame(list(zip(predictions)),columns=['Predictions'])
            path="Prediction_Output_File/Predictions.csv"
            final.to_csv("Prediction_Output_File/Predictions.csv",header=True,mode='a+') #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path




