#!/usr/bin/env python
from funcs import *
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
import pickle
import scipy.stats as st

def get_data():
    #create a empty pandas dataframe
    analysis_data = pd.DataFrame(columns=[])

    #Get the data from database and populate the dataframe
    analysis_data=query_db("credentials to retrieve data from GW-BSE database")   #Database is not released yet for public     

    #save the data in a file
    analysis_data.to_pickle("./files/db_testing.pkl")

    #read the file+add features to the dataframe+write to another file
    Add_features(["./files/db_testing_vis.pkl"],"./files/data_w_feature_vis.pkl")

get_data()

