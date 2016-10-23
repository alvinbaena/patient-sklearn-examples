'''
Created on 12/06/2016
Gini decision tree sobre datasets de pacientes 
preprocesamiento con transformacion PCA de los datos
@author: Andres Moreno B
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    np.random.seed(0)

    patient_data = pd.read_pickle("../../data/df/dataset.pickle").values
    patient_data = preprocessing.scale(patient_data)
    target = pd.read_pickle("../../data/df/target.pickle").values
    indices = np.random.permutation(len(target))

    print "Total dataset size: " + str(patient_data.shape[0])
    num_train = int(np.floor(patient_data.shape[0] * 0.75))
    print "Train set size: " + str(num_train)

    patient_data_train = patient_data[indices[0:num_train]]
    target_train = target[indices[0:num_train]]

    patient_data_test = patient_data[indices[num_train:]]
    target_test = target[indices[num_train:]]
    print "Test set size: " + str(target_test.shape[0])

    decision_tree_model = DecisionTreeClassifier(criterion='gini',
                                                 splitter='best',
                                                 max_depth=None)
    decision_tree_model.fit(patient_data_train, target_train)
    predictions = decision_tree_model.predict(patient_data_test)
    joblib.dump(decision_tree_model, '../../data/models/decision_tree_model.plk')
    np.save('../../data/predictions/decision_tree_predictions.npy', predictions)
    print "Feature importances are " + str(decision_tree_model.feature_importances_)
    truePIx = np.logical_and(target_test == 1, predictions == 1)
    trueNIx = np.logical_and(target_test == 0, predictions == 0)
    falsePIx = np.logical_and(target_test == 0, predictions == 1)
    falseNIx = np.logical_and(target_test == 1, predictions == 0)

    conf = confusion_matrix(target_test, predictions, labels=[1, 0])
    print "F1-Test decision tree model score " + str(f1_score(target_test, predictions, labels=[1, 0]))
    print conf

    train_predictions = decision_tree_model.predict(patient_data_train)
    confusion_train = confusion_matrix(target_train, train_predictions, labels=[1, 0])
    print "F1-Train logit model score " + str(f1_score(target_train, train_predictions, labels=[1, 0]))
    print "train confusion matrix"
    print confusion_train
