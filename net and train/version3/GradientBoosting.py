# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:34:10 2019

@author: lenovo
"""

import numpy as np
import pandas as pd
import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib



def testGradientBoosting(features, labels):
    
    time1 = time.time()
    
    #construct the model
    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators = 150, subsample=0.5,
                                     min_samples_split = 8, max_features = None, max_depth = 10 )
    #select k
    k = 10
    
    #reorder the trainig set
    np.random.seed(6)
    np.random.shuffle(features)
    np.random.seed(6)
    np.random.shuffle(labels)  
    
    #using k-fold crossing validation
    print('When using GradientBoosting: ')
    print('When k = ', k)
    print('When use cross_val_score: ')
    score = cross_val_score(clf, features, labels, cv = k)
    print('Accuracy: %0.5f(+/-)%0.5f' %(score.mean(), score.std()**2))
    print('When use cross_val_predict: ')
    predict = cross_val_predict(clf, features, labels, cv = k)
    print('MSE: %0.5f' % mean_squared_error(labels, predict))
    
    #train
    clf.fit(features, labels)
    time2 = time.time()
    print('total time: %0.3f' %(time2 - time1))
    
    #save model
    joblib.dump(clf, 'new_net.pkl')
#pre_treat the data
def pre_treat(features, labels): #pre_treat the data
    
    #delete some labels samples
    del_row = []
    for i in range(len(labels)):
        if int(labels[i]) in [4]:
            del_row.append(i)
    features = np.delete(features, del_row, 0)
    labels = np.delete(labels, del_row, 0)
    
    
    #change the remain labels to right order  
    for i in range(len(labels)):
        if int(labels[i]) >= 1 and int(labels[i]) <= 3:
            labels[i] = int(labels[i]) - 1
        else:
            labels[i] = int(labels[i]) - 2
    '''
    for i in range(len(labels)):
        labels[i] = int(labels[i]) - 2
    ''' 
    return features, labels

#filter none data   
def filter_none_data(features):
        #filter the None data
    for i in range(len(features)):
        non_pos = []
        for j in range(len(features[i])):
            if str(features[i][j]) == str(None):
                non_pos.append(j)
                features[i][j] = 0
                
        for k in range(10): #value iteration times
            for m in range(len(non_pos)):
                if non_pos[m]%15 == 0:  #begin position
                    features[i][non_pos[m]] = features[i][non_pos[m]+1]
                elif (non_pos[m]+1) % 15 == 0: #end position
                    features[i][non_pos[m]] = features[i][non_pos[m]-1]
                else:#medium position
                    features[i][non_pos[m]] = (features[i][non_pos[m]+1] + features[i][non_pos[m]-1]) / 2
    return features

if __name__ == '__main__':
    features = np.load('sample_expend.npy')
    labels = np.load('label_expend.npy')
    #preprocess the features and labels
    features, labels = pre_treat(features, labels)
    features = filter_none_data(features)
    #change the 
    features = features.astype(np.int32)
    
    print('GrandientBoostingClassifier\r')

    testGradientBoosting(features, labels)