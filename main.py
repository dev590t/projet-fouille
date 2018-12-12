

import numpy as np
from pca import pca_func
from sklearn.svm import SVC
from knn import *
from data_gestion import *


def svmModel(trainData, trainLabel):
    print('Train SVM...')
    svmClf = SVC(C=1.0,kernel='poly',degree=2)
    svmClf.fit(trainData, trainLabel)
    return svmClf




if __name__ == '__main__':
    trainModel = svmModel #knnClassify #svmModel
    split_data('dataset.csv','train.csv','test.csv',5555)
    
    train_data, train_label, test_data, test_label = read_data('train.csv','test.csv')
    
    pca_train,pca_test = pca_func(train_data,test_data)

    train_pca = pca_train.tolist()
    test_pca = pca_test.tolist()
    clf = trainModel(train_pca,train_label)
    print('test data ......')
    prediction_label = clf.predict(test_pca)

    prediction_label = prediction_label.tolist()

    # evaluate the precision
    count = 0
    for i in range(len(prediction_label)):
        if prediction_label[i] == test_label[i]:
            count += 1
    print(count)
    print('the right rate is:',float(count)/len(prediction_label))
    
