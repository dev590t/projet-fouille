import numpy as np

'''
    compute the value and the vector of eig
'''
def calcul_eig(train_data):
    print('calcul eig...')
    meanVals = np.mean(train_data, axis=0)
    meanRemoved = train_data-meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigvals, eigVects = np.linalg.eig(np.mat(covMat))
    return eigvals,eigVects
 
'''
    analyse data , and select some feature who have occupe 95% vars
'''
def analyse_data( eigvals, eigVects, taux=0.95):
    print('analyse data....')
    eigValInd = np.argsort(-eigvals)

    count = 0
    cov_all_score = sum(eigvals)
    sum_cov_score = 0
    for i in range(0, len(eigValInd)):
        line_cov_score = eigvals[eigValInd[i]]
        sum_cov_score += line_cov_score
        count += 1
        if sum_cov_score/cov_all_score>=taux:
            break
        print('main: {:.0f}'.format(i+1),',   sqrs: {:.2f}'.format((line_cov_score/cov_all_score*100).real),'%,  sum : {:.2f}'.format((sum_cov_score/cov_all_score*100).real),'% ')
        
    return count


'''
    cut some feature , ruduce dimension
'''
def cut_feature(train_data, test_data, num, eigvals, eigVects ):

    print('cut features ....')
    meanVals = np.mean(train_data, axis=0)
    meanVals_test = np.mean(test_data, axis=0)
    meanRemoved = train_data-meanVals
    meanRemoved_test = test_data-meanVals_test

    eigValInd = np.argsort(eigvals)
    eigValInd = eigValInd[:-(num+1):-1]
    redEigVects = eigVects[:, eigValInd]

    low_train = meanRemoved * redEigVects
    low_test = meanRemoved_test * redEigVects

    pca_train = (low_train * redEigVects.T) + meanVals
    pca_test = (low_test * redEigVects.T) + meanVals_test
    #pca_train = train_data * redEigVects
    #pca_test = test_data * redEigVects

    print('pca train',pca_train.shape,'pca test', pca_test.shape)
    return np.real(pca_train), np.real(pca_test)



def pca_func(train_data,test_data):

    eigvals, eigVects = calcul_eig(np.array(train_data).astype(np.float))
    num = analyse_data( eigvals, eigVects, 0.96)
    print('num',num)
    train_pca, test_pca = cut_feature(np.array(train_data).astype(np.float),np.array(test_data).astype(np.float),num, eigvals, eigVects )
    return train_pca, test_pca
