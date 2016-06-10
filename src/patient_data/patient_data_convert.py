'''
Created on 6/06/2016
Cambia el formato del dataset a pickles 
@author: Andres Moreno B.


'''



import pandas as pd
from os import path
import re

def convertToFloat(x):
    if isinstance(x,str):
        return float(x.replace(",","."))
    return float(x)

   
file_path = path.relpath("../../data/DatasSet5.csv")

df=pd.read_csv(file_path, header=0, delimiter=';')
df['morto']=df['morto'].replace('N' ,0)
df['morto']=df['morto'].replace('Y' ,1)
df=df.rename(columns={'morto':'target'})
print df.columns.values

sapsDf = df[['SAPSI_FIRST','SAPSI_MIN','SAPSI_MAX']]
print sapsDf

sofaDf = df[['SOFA_FIRST','SOFA_MIN','SOFA_MAX']]
print sofaDf

drop_indeces = [i for i,val in enumerate(df.columns.values) 
                if  (val.startswith('0_') or val.startswith('1_') 
                     or val.startswith('2_') or val.startswith('3_') 
                     or val.startswith('SAPSI_') or val.startswith('SOFA_')) ]


#delete periods 0 to 3, sofa and sapsI
df.drop(df.columns.values[drop_indeces],inplace=True,axis=1)
data_target=df['target']

patient_id=df['paciente']
del df['paciente']
del df['target']
print df.columns.values

df=df.applymap(lambda x: convertToFloat(x))


df.to_pickle("../../data/df/dataset.pickle")
data_target.to_pickle("../../data/df/target.pickle")
patient_id.to_pickle("../../data/df/ids.pickle")
sapsDf.to_pickle("../../data/df/saps.pickle")
sofaDf.to_pickle("../../data/df/sofa.pickle")




