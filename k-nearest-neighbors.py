# If you need:
# !pip install scikit-learn==0.23.1

# Train a k-nearest neighbors model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Input as a csv file (all the classified data to train your model):
input = 'morphology.csv'
# The names of your columns which will be used to classify tha data
columns_x = ['log(W1)', 'log(NUV)', 'n'] # , 'WISE_3.4' , 'SPIRE_250', 'GALEX_NUV', 'q', 'n'
# The name of your column with the classification (the categories to me numerical integers: e.g. 0 and 1)
column_y = 'type'


# Main

df = pd.read_csv(input)
df['log(W1)'] = np.log(df['WISE_3.4'].values)
df['log(NUV)'] = np.log(df['GALEX_NUV'].values)
df['log(250)'] = np.log(df['SPIRE_250'].values)
df = df.dropna(axis=0)
X = df[columns_x].values  #.astype(float)
y = df[column_y].values

# Normalize the data:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Split your sample
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4) # test_size=0.2 means that 20% of your sample wil be the test sample
                                                                                          # random_state is not important, but if you have to repeat it, you have to verify that you use the same

k = 4
# Train Model
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

# Predict
yhat = neigh.predict(X_test)

# Accuraccy evaluation (equal to jaccard score)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

"""

# To find the best K value

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):

    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)


    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print("The mean_accuracy for each K: ", mean_acc)

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.savefig("KNearestNeighbors.png")

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

"""
