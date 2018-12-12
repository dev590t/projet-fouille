import random
import math
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def knnClassify(trainData,trainLabel):
    print('trainning with knn:')
    knnClf = KNeighborsClassifier(n_neighbors=4);
    knnClf.fit(trainData,trainLabel)
    return knnClf

def distance_euclidienne(data1,data2):
    """Computes the Euclidian distance between data1 and data2.
  
  Args:
    data1: a list of numbers: the coordinates of the first vector.
    data2: a list of numbers: the coordinates of the second vector (same length as data1).

  Returns:
    The Euclidian distance: sqrt(sum((data1[i]-data2[i])^2)).
  """

    return (sum([(data1[x]-data2[x])**2 for x in range(len(data1))]))**0.5


def k_nearest_neighbors(x,points,dist_function,k):
    """Returns the indices of the k elements of points that are closest to x.
    
  Args:
    x: a list of numbers: a N-dimensional vector.
    points: a list of list of numbers: a list of N-dimensional vectors.
    dist_function: a function taking two N-dimensional vectors as
        arguments and returning a number. Just like simple_distance.
    k: an integer. Must be smaller or equal to the length of "points".

  Returns:
    A list of integers: the indices of the k elements of "points" that are
    closest to "x" according to the distance function dist_function.
    IMPORTANT: They must sorted by distance: nearest neighbor first.
  """
    distances = [ (i,dist_function(x,points[i])) for i in range(len(points))] 
    sortList = sorted(distances,key=lambda x:x[1])
    return [sortList[x][0] for x in range(k)]



def image_number(x,train_x,train_y,dist_function,k):
    """Predicts the number of image between 0 and 9, using KNN.
    
  Args:
    x: A list of floats representing a data point that we want to diagnose.
    train_x: A list of list of floats representing the data points of
        the training set.
    train_y: A list of booleans representing the classification of
        the training set: between 0 and 9. Same length as 'train_x'.
    dist_function: Same as in k_nearest_neighbors().
    k: Same as in k_nearest_neighbors().

  Returns:
    A boolean: number between 0 and 9
  """

    k_indexes = k_nearest_neighbors(x,train_x,dist_function,k)
    k_bools = [ train_y[i] for i in k_indexes]
    flag = False
    falses = 0
    trues = 0
    for i in k_bools:
        if i:
            trues += 1
        else:
            falses += 1
    if trues > falses:
        flag = True
    return flag


def eval_knn_classifier(train_x,train_y,test_x,test_y,classifier,dist_function,k):
    total = len(test_x)
    bools = [ classifier(test_x[i],train_x,train_y,dist_function,k) for i in range(total)]
    count = 0
    for i in range(total):
        if bools[i] != test_y[i]:
            count += 1

    #print(k, float(count/total))
    return float(count/total)


def get_weighted_dist_function(train_x,train_y):
    size = len(train_x[0])
    f_means = [ sum(train_x[:][j])/len(train_x[:][j]) for j in range(size)]
    f_sqrt = []
    for j in range(size):
        sum_var = 0.0
        for i in range(len(train_x)):
            var = (train_x[i][j]-f_means[j])**2
            sum_var += var
        f_sqrt.append(sum_var**0.5)

    def weighted_distance(data1,data2):
        return (sum([(data1[x]-data2[x])**2/(f_sqrt[x]+1) for x in range(len(data1))]))**0.5

    return weighted_distance


