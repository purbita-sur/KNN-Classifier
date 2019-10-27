import operator
import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self,k):
        self.k=k

    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print("Training done")

    def predict(self,X_test):
        distance={}
        counter=0
        n=self.X_train.shape[1]
        classification=np.array([],dtype='int')
        for i in X_test:
            p=0
            for j in self.X_train:
                p=p+np.sum((i-j) ** 2)
                p=p**1/2
                distance[counter]=p
                p=0
                counter=counter+1
            distance=sorted(distance.items(),key=operator.itemgetter(1))
            m=self.classify(distance[0:self.k])
            classification=np.append(classification,m)
            counter=0
            distance={}
        return classification
    def classify(self,distance):
        label=[]
        for i in distance:
            label.append(self.y_train[i[0]])

        return Counter(label).most_common()[0][0]