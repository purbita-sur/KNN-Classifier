import numpy as np
import pandas as pd
from KNearestNeighbor import KNearestNeighbors

data=pd.read_csv("Social_Network_Ads.csv")
X=data.iloc[:,2:4].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

print(X_train)

#An object of KNN
knn=KNearestNeighbors(k=5)

knn.fit(X_train,y_train)

def predict_new():
    result = knn.predict(X_test).tolist()
    for i in range(0, len(result)):
        if result[i] == 0:
            result[i]='Will not Purchase'
        elif result[i] == 1:
            result[i] = 'Will purchase'
        print(result[i])
predict_new()