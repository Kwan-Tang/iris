from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

iris = load_iris()

dataset = pd.DataFrame(data=iris.data,columns=iris.feature_names)
dataset['target'] = iris.target
dataset['target'].replace({0:iris.target_names[0],1:iris.target_names[1],2:iris.target_names[2]},inplace=True)

X_train,X_test,y_train,y_test = train_test_split(dataset.iloc[:,0:4],dataset['target'])
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

print("\n\n\n***Accuracy score: " + "{:.2%}".format(knn.score(X_test,y_test)) + "\n\n\n")
arr = []
for i in range(4):
    arr.append(float(input("Please enter " + iris.feature_names[i] + ": ")))
arr = np.array([arr])

print("\n\n\nPrediction is " + knn.predict(arr)[0]+"\n\n\n")
