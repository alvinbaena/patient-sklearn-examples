'''
Created on 11/06/2016
Busca una clasificacion sobre datasets de pacientes usando SVM con kernel rbf
preprocesamiento con transformacion PCA de los datos
@author: Andres Moreno B
'''
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from validation_tools import util

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

    # keep 95% of variance
    pca = decomposition.PCA(n_components=0.95)
    pca.fit(patient_data_train)

    print "To keep at least 95% of variance #of components are " + str(len(pca.explained_variance_ratio_))
    print "Explained variance is " + str(np.sum(pca.explained_variance_ratio_))

    print patient_data_train.shape

    X = pca.transform(patient_data_train)
    print X.shape
    X_test = pca.transform(patient_data_test)

    exploreC = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    exploreGamma = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    param_grid = [

        {'C': exploreC, 'gamma': exploreGamma, 'kernel': ['rbf']},
    ]
    best_err = float('-inf')
    best_c = 1
    for times in range(0, 3):
        svm_rbf_model = svm.SVC(C=1, kernel='rbf', shrinking=True,
                                probability=False, cache_size=2000,
                                verbose=False, max_iter=-1,
                                decision_function_shape='ovr', random_state=0)

        clf = GridSearchCV(svm_rbf_model, param_grid, cv=5,
                           scoring='f1')
        clf.fit(X, target_train)
        print(clf.best_params_)
        exploreC = util.find_new_explore_c(exploreC, clf.best_params_['C'])
        exploreGamma = util.find_new_explore_c(exploreGamma, clf.best_params_['gamma'])
        param_grid = [

            {'C': exploreC, 'gamma': exploreGamma, 'kernel': ['rbf']},
        ]

    svm_rbf_model = svm.SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], kernel='rbf', shrinking=True,
                            probability=False, cache_size=2000,
                            verbose=False, max_iter=-1,
                            decision_function_shape='ovr', random_state=0)
    svm_rbf_model.fit(X, target_train)
    joblib.dump(svm_rbf_model, '../../data/models/svm_rbf_model.plk')
    predictions = svm_rbf_model.predict(X_test)
    np.save('../../data/predictions//svm_rbf_mode_predictions.npy', predictions)

    truePIx = np.logical_and(target_test == 1, predictions == 1)
    trueNIx = np.logical_and(target_test == 0, predictions == 0)
    falsePIx = np.logical_and(target_test == 0, predictions == 1)
    falseNIx = np.logical_and(target_test == 1, predictions == 0)

    conf = confusion_matrix(target_test, predictions, labels=[1, 0])
    print "F1-Test rbf kernel model score " + str(f1_score(target_test, predictions, labels=[1, 0]))
    print "test confusion matrix"
    print conf

    train_predictions = svm_rbf_model.predict(X)
    confusion_train = confusion_matrix(target_train, train_predictions, labels=[1, 0])

    print "F1-Train rbf kernel model score " + str(f1_score(target_train, train_predictions, labels=[1, 0]))
    print "train confusion matrix"
    print confusion_train
