import pandas as pd
import sys
from funcs import *
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

api_key=<MP api key>

def diag_eps(row,column):
    try:
        return [row[column][0][0], row[column][1][1], row[column][2][2]]
    except:
        return float('nan')

def check_icsd(row,column):
    try:
        if len(row[column])==0:
            return False 
        else:
            return True
    except:
        return float('nan')

def get_mpdata():
    mpdr = MPDataRetrieval(api_key)
    df_1 = mpdr.get_dataframe(criteria={"band_gap": {"$gt": 0.1}, "diel.e_electronic": {"$exists": True}}, 
        properties=['material_id','diel.e_electronic','band_gap','structure','pretty_formula','elements', 'icsd_ids'],index_mpid=False)
    df_2 = pd.read_pickle("./files/mp_effmass.pkl")
    df = pd.merge(df_1, df_2, on="material_id")
    df["dielectric_constant"]=df.apply (lambda row: diag_eps(row,"diel.e_electronic"), axis=1)
    df["icsd"]=df.apply (lambda row: check_icsd(row,"icsd_ids"), axis=1)
    df=df.drop(['diel.e_electronic'], axis=1)
    df=df.rename(columns={"band_gap": "dft_gap"}) 
    df=df.rename(columns={"pretty_formula": "formula"}) 


get_mpdata()
Add_features(["./files/mpdata_master.pkl"],"./files/mpdata_wf.pkl")
df = pd.read_pickle("./files/mpdata_wf.pkl")

