'''
Crea el plot de area bajo la curva para los modelos 
Created on 12/06/2016

@author: Andres Moreno B
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc




if __name__ == '__main__':
    np.random.seed(0)

    patient_data=pd.read_pickle("../../data/df/dataset.pickle").values
    patient_data=preprocessing.scale(patient_data)
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
    
    all_predictions=dict()
   
    saps_all_model_predictions=np.load('../../data/predictions/saps_all_model_predictions.npy')
    saps_init_model_predictions=np.load('../../data/predictions/saps_init_model_predictions.npy')
    all_predictions['Initial SAPS model']=saps_init_model_predictions

    decision_tree_predictions=np.load('../../data/predictions/decision_tree_predictions.npy')
    all_predictions['Gini decision tree']=decision_tree_predictions
    logit_model_predictions=np.load('../../data/predictions/logit_model_predictions.npy')
    all_predictions['Logistic regression']=logit_model_predictions
    
    svm_model_predictions=np.load('../../data/predictions/svm_model_predictions.npy')
    all_predictions['Linear SVM']=svm_model_predictions
    svm_poly2_model_predictions=np.load('../../data/predictions/svm_poly2_model_predictions.npy')
    svm_poly3_model_predictions=np.load('../../data/predictions/svm_poly3_model_predictions.npy')
    all_predictions['Polynomial n=3 SVM']=svm_poly3_model_predictions
    svm_poly4_model_predictions=np.load('../../data/predictions/svm_poly4_model_predictions.npy')
    svm_poly5_model_predictions=np.load('../../data/predictions/svm_poly5_model_predictions.npy')
    svm_rbf_mode_predictions=np.load('../../data/predictions/svm_rbf_mode_predictions.npy')
    all_predictions['RBF  SVM']=svm_rbf_mode_predictions
    
    
    
    plt.title('Receiver Operating Characteristic')
    
    for key in all_predictions.keys():
        false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, all_predictions[key],pos_label=1)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate,  label='AUC '+key+'= %0.4f'% roc_auc)
    
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],color='black', linestyle='dashed')
    plt.xlim([-0.01,1.2])
    plt.ylim([-0.01,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
    
    
    
    
    
    

    