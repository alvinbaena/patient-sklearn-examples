'''
Created on 7/06/2016
Busca una regresion lineal sobre datasets de pacientes usando Logistic Regression
preprocesamiento con transformacion PCA de los datos
@author: Andres Moreno B
'''
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from validation_tools import util





if __name__ == '__main__':
    np.random.seed(0)

    patient_data=pd.read_pickle("../../data/df/dataset.pickle").values
    target=pd.read_pickle("../../data/df/target.pickle").values
    indices = np.random.permutation(len(target))

    print "Total dataset size: "+str(patient_data.shape[0])
    num_train=int(np.floor(patient_data.shape[0]*0.75))
    print "Train set size: "+str(num_train)


    patient_data_train=patient_data[indices[0:num_train]]
    target_train=target[indices[0:num_train]]

    patient_data_test=patient_data[indices[num_train:]]
    target_test=target[indices[num_train:]]
    print "Test set size: "+str(target_test.shape[0])


    #keep 95% of variance
    pca = decomposition.PCA(n_components=0.95)
    pca.fit(patient_data_train)

    print "To keep at least 95% of variance #of components are "+str(len(pca.explained_variance_ratio_))
    print "Explained variance is "+str(np.sum(pca.explained_variance_ratio_))

    print patient_data_train.shape
    
    X=pca.transform(patient_data_train)
    print X.shape
    X_test=pca.transform(patient_data_test)
    
    exploreC=[0.0001,0.001,0.01,0.1,1,10]
    for i in range(1,5):
        logitmodel=linear_model.LogisticRegressionCV(exploreC, fit_intercept=True, cv=5,  penalty='l2', dual=False,  solver='liblinear',  n_jobs=-1, verbose=0, refit=True, random_state=0,  scoring='f1_weighted')
    
        logitmodel.fit(X,target_train)
        predictions= logitmodel.predict(X_test)
        scores=logitmodel.scores_[1]
        best_val=logitmodel.C_
        print "CV averages for values "+str(exploreC)+" are:"+str(np.average(scores,0))
        print "Best C is"+str(logitmodel.C_)
        exploreC=util.find_new_explore_c(exploreC, best_val)
    
    
    print logitmodel.score(X_test, target_test)
    truePIx=np.logical_and(target_test==1,predictions==1)
    trueNIx=np.logical_and(target_test==0,predictions==0)
    falsePIx=np.logical_and(target_test==0,predictions==1)
    falseNIx=np.logical_and(target_test==1,predictions==0)
    print np.sum(truePIx)+np.sum(trueNIx)+np.sum(falsePIx)+np.sum(falseNIx)
    
    conf=confusion_matrix(target_test, predictions, labels=[1,0])
    print conf
    
    labelsFig=np.array([None]*X_test.shape[0])
   
    labelsFig[truePIx]='lime'
   
    labelsFig[trueNIx]='green'
    labelsFig[falsePIx]='red'
    labelsFig[falseNIx]='black'

   
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=labelsFig.tolist())
    plt.show()
    plt.clf()
    
    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    f, ax = plt.subplots(figsize=(8, 6))
    grid=np.c_[xx.ravel(),yy.ravel()]
    grid=np.c_[grid,np.zeros((xx.ravel().shape[0],10))]
    probs = logitmodel.predict_proba(grid)[:, 1].reshape(xx.shape)
    print probs
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                      vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    ax.scatter(X_test[:,0], X_test[:, 1], c=labelsFig.tolist(), s=50,cmap="RdBu", vmin=-5, vmax=5,edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
       xlim=(-5, 5), ylim=(-5, 5),
       xlabel="$X_1$", ylabel="$X_2$")
    #plt.show()
    
    
     








