'''
Created on 10/06/2016
Busca una regresion lineal sobre datos SAPS usando Logistic Regression
@author: Andres Moreno B


'''
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from validation_tools import util
from sklearn.externals import joblib



if __name__ == '__main__':
    np.random.seed(0)

    saps_data=pd.read_pickle("../../data/df/saps.pickle").values
    target=pd.read_pickle("../../data/df/target.pickle").values
    indices = np.random.permutation(len(target))

    print "Total dataset size: "+str(saps_data.shape[0])
    num_train=int(np.floor(saps_data.shape[0]*0.75))
    print "Train set size: "+str(num_train)

    #Model with only initial saps and logarithm of initial saps score
    initSaps=saps_data[:,1]
    logInitSaps=np.log1p(initSaps)
    initSaps=np.transpose(np.vstack((initSaps,logInitSaps)))
    saps_data_train=initSaps[indices[0:num_train]]
    target_train=target[indices[0:num_train]]

    saps_data_test=initSaps[indices[num_train:]]
    target_test=target[indices[num_train:]]
    print "Test set size: "+str(target_test.shape[0])

    
    
    exploreC=[0.0001,0.001,0.01,0.1,1,10]
    for i in range(0,5):
        logitmodelInitSAPS=linear_model.LogisticRegressionCV(exploreC, fit_intercept=True, cv=5,  penalty='l2', dual=False,  solver='liblinear',  n_jobs=-1, verbose=0, refit=True, random_state=0,  scoring='f1_weighted')
        
    
        logitmodelInitSAPS.fit(saps_data_train,target_train)
 
        predictions= logitmodelInitSAPS.predict(saps_data_test)
        scores=logitmodelInitSAPS.scores_[1]
        best_val=logitmodelInitSAPS.C_
        print "CV averages for values "+str(exploreC)+" are:"+str(np.average(scores,0))
        print "Best C is"+str(best_val)
        
        exploreC=util.find_new_explore_c(exploreC, best_val)
         
        
    joblib.dump(logitmodelInitSAPS, '../../data/models/saps_init_model.plk') 
    train_predictions= logitmodelInitSAPS.predict(saps_data_train)
    
    truePIx=np.logical_and(target_train==1,train_predictions==1)
    trueNIx=np.logical_and(target_train==0,train_predictions==0)
    falsePIx=np.logical_and(target_train==0,train_predictions==1)
    falseNIx=np.logical_and(target_train==1,train_predictions==0)
    
    
    
    labelsFig=np.array([None]*train_predictions.shape[0])
    
    labelsFig[truePIx]='lime'
    labelsFig[trueNIx]='green'
    labelsFig[falsePIx]='red'
    labelsFig[falseNIx]='red'
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax =Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
     
    ax.scatter(np.arange(len(train_predictions)), train_predictions, saps_data_train[:, 0] ,  c=labelsFig.tolist())
    plt.show()
    
    print logitmodelInitSAPS.score(saps_data_test, target_test)
    truePIx=np.logical_and(target_test==1,predictions==1)
    trueNIx=np.logical_and(target_test==0,predictions==0)
    falsePIx=np.logical_and(target_test==0,predictions==1)
    falseNIx=np.logical_and(target_test==1,predictions==0)
    print np.sum(truePIx)+np.sum(trueNIx)+np.sum(falsePIx)+np.sum(falseNIx)
     
    conf_initial_scores=confusion_matrix(target_test, predictions, labels=[1,0])
    print conf_initial_scores
     
    labelsFig=np.array([None]*saps_data_test.shape[0])
    
    labelsFig[truePIx]='lime'
    
    labelsFig[trueNIx]='green'
    labelsFig[falsePIx]='red'
    labelsFig[falseNIx]='red'

    
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax =Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
     
    ax.scatter(np.arange(len(predictions)), predictions,saps_data_test[:, 0] ,  c=labelsFig.tolist())
    plt.show()
    
    

    #logit model with all saps data
    saps_data_train=saps_data[indices[0:num_train]]
    saps_data_test=saps_data[indices[num_train:]]
    exploreC=[0.0001,0.001,0.01,0.1,1,10]
    for i in range(0,5):
        logitmodelAllSAPS=linear_model.LogisticRegressionCV(exploreC, fit_intercept=True, cv=5,  penalty='l2', dual=False,  solver='liblinear',  n_jobs=-1, verbose=0, refit=True, random_state=0,  scoring='f1_weighted')
        
    
        logitmodelAllSAPS.fit(saps_data_train,target_train)
 
        predictions= logitmodelAllSAPS.predict(saps_data_test)
        scores=logitmodelAllSAPS.scores_[1]
        best_val=logitmodelAllSAPS.C_
        print "CV averages for values "+str(exploreC)+" are:"+str(np.average(scores,0))
        print "Best C is"+str(best_val)
        
        exploreC=util.find_new_explore_c(exploreC, best_val)

    joblib.dump(logitmodelAllSAPS, '../../data/models/saps_all_model.plk')
    train_predictions= logitmodelAllSAPS.predict(saps_data_train)
    
    truePIx=np.logical_and(target_train==1,train_predictions==1)
    trueNIx=np.logical_and(target_train==0,train_predictions==0)
    falsePIx=np.logical_and(target_train==0,train_predictions==1)
    falseNIx=np.logical_and(target_train==1,train_predictions==0)
    
    
    
    labelsFig=np.array([None]*train_predictions.shape[0])
    
    labelsFig[truePIx]='lime'
    labelsFig[trueNIx]='green'
    labelsFig[falsePIx]='red'
    labelsFig[falseNIx]='red'
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax =Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
     
    ax.scatter(np.arange(len(train_predictions)), train_predictions, saps_data_train[:, 0] ,  c=labelsFig.tolist())
    plt.show()
    
    print logitmodelAllSAPS.score(saps_data_test, target_test)
    truePIx=np.logical_and(target_test==1,predictions==1)
    trueNIx=np.logical_and(target_test==0,predictions==0)
    falsePIx=np.logical_and(target_test==0,predictions==1)
    falseNIx=np.logical_and(target_test==1,predictions==0)
    print np.sum(truePIx)+np.sum(trueNIx)+np.sum(falsePIx)+np.sum(falseNIx)
     
    conf_all_scores=confusion_matrix(target_test, predictions, labels=[1,0])
    print conf_all_scores
     
    labelsFig=np.array([None]*saps_data_test.shape[0])
    
    labelsFig[truePIx]='lime'
    
    labelsFig[trueNIx]='green'
    labelsFig[falsePIx]='red'
    labelsFig[falseNIx]='red'

    
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax =Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
     
    ax.scatter(np.arange(len(predictions)), predictions,saps_data_test[:, 0] ,  c=labelsFig.tolist())
    plt.show()
        
    



