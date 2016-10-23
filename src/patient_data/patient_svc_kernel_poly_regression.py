'''
Created on 11/06/2016
Busca una clasificacion sobre datasets de pacientes usando SVM con kernel polinomial
preprocesamiento con transformacion PCA de los datos
@author: Andres Moreno B
'''
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib
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
    poly = [2, 3, 4, 5]
    best_err = float('-inf')
    best_c = 1
    best_poly = 0

    for j in range(0, len(poly)):
        for times in range(0, 3):

            results = np.zeros(len(exploreC))
            for i in range(0, len(exploreC)):
                svm_poly_model = svm.SVC(exploreC[i], kernel='poly',
                                         degree=poly[j], shrinking=True,
                                         probability=False, cache_size=2000,
                                         verbose=False, max_iter=-1,
                                         decision_function_shape='ovr', random_state=0)

                scores_cv = cross_validation.cross_val_score(svm_poly_model, X,
                                                             target_train,
                                                             scoring='f1_weighted', cv=5,
                                                             n_jobs=-1)

                results[i] = float(scores_cv.mean())
                print "run for C=" + str(exploreC[i]) + ", poly=" + str(poly[j]) + " is " + str(results[i])
            print results
            best_val = np.max(results)
            print "Best result of iteration was " + str(best_val)
            results = results.tolist()
            index_val = results.index(best_val)
            if best_val > best_err:
                best_err = best_val
                best_c = exploreC[index_val]
                best_poly = poly[j]
            print "CV averages for values " + str(exploreC) + " are:" + str(results)
            print "Best C is" + str(exploreC[index_val])
            exploreC = util.find_new_explore_c(exploreC, exploreC[index_val])

        svm_poly_model = svm.SVC(best_c, kernel='poly',
                                 degree=poly[j], shrinking=True,
                                 probability=False, cache_size=2000,
                                 verbose=False, max_iter=-1,
                                 decision_function_shape=None, random_state=0)
        svm_poly_model.fit(X, target_train)

        predictions = svm_poly_model.predict(X_test)
        joblib.dump(svm_poly_model, '../../data/models/svm_poly' + str(poly[j]) + '_model.plk')
        np.save('../../data/predictions/svm_poly' + str(poly[j]) + '_model_predictions.npy', predictions)

        truePIx = np.logical_and(target_test == 1, predictions == 1)
        trueNIx = np.logical_and(target_test == 0, predictions == 0)
        falsePIx = np.logical_and(target_test == 0, predictions == 1)
        falseNIx = np.logical_and(target_test == 1, predictions == 0)

        conf = confusion_matrix(target_test, predictions, labels=[1, 0])
        print "F1-Test linear kernel model score poly" + str(poly[j]) + " " + str(
            f1_score(target_test, predictions, labels=[1, 0]))
        print "Confusion poly " + str(poly[j]) + " " + str(conf)

        train_predictions = svm_poly_model.predict(X)
        confusion_train = confusion_matrix(target_train, train_predictions, labels=[1, 0])
        print "F1-Train linear kernel model score poly" + str(poly[j]) + " " + str(
            f1_score(target_train, train_predictions, labels=[1, 0]))
        print confusion_train

    print "Best c " + str(best_c)
    print "Best poly " + str(best_poly)
