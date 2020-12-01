#Import packages relative to computation and graphing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Import packages relative to the dataset and regression
from sklearn import datasets
from sklearn.cluster import KMeans 
from sklearn import metrics
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist 


#Question2
#Getting the dataset
Wine = datasets.load_wine()
Wine = pd.DataFrame(Wine.data, columns = Wine.feature_names)
Iris = datasets.load_iris()
Iris = pd.DataFrame(Iris.data, columns = Iris.feature_names)


#define a function for elow heuristic
def elbow_heuristic(X):
    #set up the empty for res=inertias, distort=distortion.
    distort = [] 
    res_inertia = [] 
    result1 = {} 
    result2 = {} 
    K = range(1,10)
    for k in K: 
    #Fitting the model, and this loop is focus on the inertia
        k_Model = KMeans(n_clusters=k).fit(X)
        k_Model.fit(X)

        res_cdist=cdist(X, k_Model.cluster_centers_, 'euclidean')
        res_distort=sum(np.min(res_cdist,axis=1)) / X.shape[0]
        distort.append(res_distort)
        
        res_inertia.append(k_Model.inertia_)
        result1[k] = res_distort
        result2[k] = k_Model.inertia_

        
    print("the value of distortion is :")
    print(result2.items())
    print('\n')
    print("the value of inertia is:")
    print(result2.items())


 
    plt.plot(K, res_inertia, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Inertia') 
    plt.show()

elbow_heuristic(Wine)

print('----------')
elbow_heuristic(Iris)
#according to the turning point of the graph and the value of the inertias, we can say that 2 is the optimum for Wine and Iris.

