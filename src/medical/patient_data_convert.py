'''
Created on 6/06/2016
Cambia el formato del dataset a pickles 
@author: Andres Moreno B.


'''

from os import path

import pandas as pd


def convert_to_float(x):
    if isinstance(x, str):
        return float(x.replace(",", "."))
    return float(x)


# file_path = path.relpath("../../data/DataSet1_med_mcghr.csv")
# file_path = path.relpath("../../data/DataSet2_med_mcgkghr.csv")
# file_path = path.relpath("../../data/DataSet3_med_uhr.csv")
# file_path = path.relpath("../../data/DataSet1_icd9_mcghr.csv")
# file_path = path.relpath("../../data/DataSet2_icd9_mcgkghr.csv")
# file_path = path.relpath("../../data/DataSet3_icd9_uhr.csv")
# file_path = path.relpath("../../data/DataSet1_icd9_med.csv")
# file_path = path.relpath("../../data/DataSet1_add_mcghr.csv")
# file_path = path.relpath("../../data/DataSet2_add_mcgkghr.csv")
# file_path = path.relpath("../../data/DataSet3_add_uhr.csv")
file_path = path.relpath("../../data/DataSet4_add_all.csv")

df = pd.read_csv(file_path, header=0, delimiter=';')
df = df.dropna(0, 'any')
df['morto'] = df['morto'].replace('N', 0)
df['morto'] = df['morto'].replace('Y', 1)
df = df.rename(columns={'morto': 'target'})
print df.columns.values

drop_indeces = [i for i, val in enumerate(df.columns.values)
                if (val.startswith('0_') or val.startswith('1_')
                    or val.startswith('2_') or val.startswith('3_'))]

# delete periods 0 to 3, sofa and sapsI
df.drop(df.columns.values[drop_indeces], inplace=True, axis=1)
data_target = df['target']

patient_id = df['paciente']
del df['paciente']
del df['target']
print df.columns.values

df = df.applymap(lambda x: convert_to_float(x))

df.to_pickle("../../data/df/medical.pickle")
data_target.to_pickle("../../data/df/target.pickle")
patient_id.to_pickle("../../data/df/ids.pickle")
