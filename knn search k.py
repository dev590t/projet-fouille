from data_gestion import *
from knn import *
from matplotlib.pyplot import *
from main import *

def load_data(train,test):
    train_data, train_label, test_data, test_label = read_data(train,test)

    # centralize, reduce data
    pca_train,pca_test = pca_func(train_data,test_data)

    train_pca = pca_train.tolist()
    test_pca = pca_test.tolist()
    return train_pca,train_label, test_pca, test_label


'''
    ###   Exercice 6 ###

    La meilleur K est 5 , taux d'erreur est 0.06484641
'''

def sampled_range(mini,maxi,num):
    if not num:
        return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi-lmini)/(num-1)
    out = [x for x in set([int(math.exp(lmini+i*ldelta)) for i in range(num)])]
    out.sort()
    return out

def test_few_k(train_x,train_y,dist_function):
    list_k = sampled_range(2,50,11)
    values = []
    j = 0
    sfolder = StratifiedKFold(n_splits = 10,random_state=0,shuffle=False)
    for train , test in sfolder.split(train_x,train_y):
        train_X = [train_x[i] for i in train]
        train_Y = [train_y[i] for i in train]
        test_X  = [train_x[i] for i in test]
        test_Y  = [train_y[i] for i in test]
        values.append(eval_knn_classifier(train_X,train_Y,test_X,test_Y,image_number,dist_function,list_k[j]))
        j += 1


    return list_k,values

def test_few_k2(train_pca, test_pca):
    list_k = sampled_range(2,50,11)    
    error = 1 - knn.score(xtest, ytest)

if __name__ == '__main__':
    train_pca,train_label, test_pca, test_label = load_data('train.csv','test.csv')
    clf = knnClassify(train_pca,train_label)
    
    # k_tested,error_rate = test_few_k(train_pca, test_pca, distance_euclidienne)

    # list_k = sampled_range(1,50,3)
    #list_k = range(1,10)
    list_k = range(1,50,5)
    error_rate = []
    for i in range(len(list_k)):
        clf.set_params( n_neighbors = list_k[i])
        error = 1 - clf.score(test_pca, test_label)
        error_rate.append(error)
                        
    title("valeur de plusieurs K")
    plot(list_k,error_rate)
    xlabel("valeur K")
    ylabel("taux erreur")
    show()
    best_k = list_k[error_rate.index(min(error_rate))]
    print("best k:", best_k)
