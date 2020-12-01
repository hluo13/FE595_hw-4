#Import packages relative to computation and graphing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Import packages relative to the dataset and regression
from sklearn import datasets
from sklearn import linear_model
from sklearn.cluster import KMeans 
from sklearn import metrics
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist 


#Question 1
#Getting the dataset,and print DESCR in order to get the description ofthe dataset.
boston = datasets.load_boston()
print(boston.DESCR)
#We can find that there are 13 variables and 506 rows.

#First, I need to convert this to dataframe. and add a column [Price]. 
boston_new = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_new['PRICE'] = boston.target

#Now I make x be dependent variable and y be independent variable to make linear regression
x = boston_new.drop('PRICE', axis = 1)
y = boston_new['PRICE']
reg=linear_model.LinearRegression()
reg.fit(x,y)
table=reg.coef_

#name the column and add a column for the nams.
table = pd.DataFrame(table)
table.columns=["coef_"]
table.insert(0, "Name", boston.feature_names, True)

#sorted them and print it out
table=table.sort_values(by = "coef_",ascending=False) 
table=table.values.tolist()
print(table)


