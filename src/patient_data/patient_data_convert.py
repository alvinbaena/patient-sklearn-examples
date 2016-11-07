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


file_path = path.relpath("../../data/DataSet1_med.csv")

df = pd.read_csv(file_path, header=0, delimiter=';')
df = df.dropna(0, 'any')
df['morto'] = df['morto'].replace('N', 0)
df['morto'] = df['morto'].replace('Y', 1)
df = df.rename(columns={'morto': 'target'})
print df.columns.values

# sapsDf = df[['sapsi_first', 'sapsi_min', 'sapsi_max']]
# print sapsDf
#
# sofaDf = df[['sofa_first', 'sofa_min', 'sofa_max']]
# print sofaDf

medDf = df[['med_dur_7', 'med_dur_6', 'med_dur_5', 'med_dur_4']]
medRecDf = df[['med_rec_7', 'med_rec_6', 'med_rec_5', 'med_rec_4']]
medFreqDf = df[['med_freq_7', 'med_freq_6', 'med_freq_5', 'med_freq_4']]
medDoseDf = df[['med_dose_7', 'med_dose_6', 'med_dose_5', 'med_dose_4']]

drop_indeces = [i for i, val in enumerate(df.columns.values)
                if (val.startswith('0_') or val.startswith('1_')
                    or val.startswith('2_') or val.startswith('3_')
                    or val.startswith('med_dur') or val.startswith('med_rec')
                    or val.startswith('med_freq') or val.startswith('med_dose'))]

# delete periods 0 to 3, sofa and sapsI
df.drop(df.columns.values[drop_indeces], inplace=True, axis=1)
data_target = df['target']

patient_id = df['paciente']
del df['paciente']
del df['target']
print df.columns.values

df = df.applymap(lambda x: convert_to_float(x))

df.to_pickle("../../data/df/dataset.pickle")
data_target.to_pickle("../../data/df/target.pickle")
patient_id.to_pickle("../../data/df/ids.pickle")
# sapsDf.to_pickle("../../data/df/saps.pickle")
# sofaDf.to_pickle("../../data/df/sofa.pickle")
medDf.to_pickle("../../data/df/dataset_med.pickle")
