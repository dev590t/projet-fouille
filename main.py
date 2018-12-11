import csv
import random
import numpy as np
from pca import pca_func
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

'''
    split dataset to 2 part, 70% is for train , and other 30% for test
'''
def split_data(in_file,out_file1,out_file2,seed = None):

    source = open(in_file,'r')
    reader = csv.reader(source)
    allData = list(reader)
    #print (len(allData[0]),allData[1])

    out_train = open(out_file1,'w')
    writer_train = csv.writer(out_train)

    out_test = open(out_file2,'w')
    writer_test = csv.writer(out_test)

    if seed:
        random.seed(seed)
    else:
        seed = random.randint(1,900)
        random.seed(seed)

    train_indexes = []
    test_indexes = []

    for i in range(0,len(allData)):
        if(random.random()>0.3):
            train_indexes.append(i)
        else:
            test_indexes.append(i)

    for index in train_indexes:
        writer_train.writerow(allData[index])

    for index in test_indexes:
        writer_test.writerow(allData[index])

    out_train.close()
    out_test.close()
    source.close()

'''
    load data, and get train_data, train_label, test_data, test_label
'''
def read_data(train_file,test_file):

    train = open(train_file,'r')
    reader_tr = csv.reader(train)
    train_set = list(reader_tr)

    test = open(test_file,'r')
    reader_te = csv.reader(test)
    test_set = list(reader_te)

    train_data = [train_set[i][1:] for i in range(1,len(train_set))]
    train_label = [train_set[i][0] for i in range(1,len(train_set))]

    test_data = [test_set[i][1:] for i in range(len(test_set))]
    test_label = [test_set[i][0] for i in range(len(test_set))]

    train.close()
    test.close()

    return train_data, train_label, test_data, test_label


def svmModel(trainData, trainLabel):
    print('Train SVM...')
    svmClf = SVC(C=1.0,kernel='poly',degree=2)
    svmClf.fit(trainData, trainLabel)
    return svmClf

def knnClassify(trainData,trainLabel):
    print('trainning with knn:')
    knnClf = KNeighborsClassifier();
    knnClf.fit(trainData,trainLabel)
    return knnClf




if __name__ == '__main__':
    trainModel = knnClassify #svmModel
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
    
