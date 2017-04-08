#Sir, the provided distance dunction wasn't very clear but I hae applied the formulae as the max/min of the mod of training data points with the input i.e the testing data points in order to find the accuracy. 
#accuracy has been returned by the knn function by the name vote_result.
#accuracy for the given distance function is 73% while for the euclidean_distance is 93%.
#to check accuracy for euclidean_disance, just uncomment that part(lines 16,17) and comment out new_distance part(lines 18,19)

import numpy as np
from collections import Counter
import pandas as pd
import random
from math import sqrt

def knn(data,predict,k):
    distances=[]
    for group in data:
        for features in data[group]:
            # euclidean_distance = sqrt((features[0] - predict[0]) ** 2 + (features[1] - predict[1]) ** 2)
            # distances.append([euclidean_distance,group])
            new_distance = (max(sqrt((features[0]**2)+(features[1]**2)),sqrt((features[2]**2)+(features[3]**2)),sqrt((predict[0]**2)+(predict[1]**2)),sqrt((predict[2]**2)+(predict[3]**2))))/(1+min(sqrt((features[0]**2)+(features[1]**2)),sqrt((features[2]**2)+(features[3]**2)),sqrt((predict[0]**2)+(predict[1]**2)),sqrt((predict[2]**2)+(predict[3]**2))))
            distances.append([new_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


df=pd.read_csv('iris.data.txt')
df.replace('Iris-setosa',1,inplace=True)
df.replace('Iris-versicolor',2,inplace=True)
df.replace('Iris-virginica',3,inplace=True)
full_data = df.astype(float).values.tolist()

#iris-sentosa = 1
#iris-versicolor = 2
#iris-virginicia = 3

random.shuffle(full_data)
test_size=0.2
train_set={1:[], 2:[], 3:[]}
test_set={1:[], 2:[], 3:[]}
train_data = full_data[:-int(test_size*len(full_data))] # 80% training data
test_data = full_data[-int(test_size*len(full_data)):] # 20% testing data



for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

total=0
correct=0


for group in test_set:
    for data in test_set[group]:
        vote = knn(train_set, data, 5) #defining k as 5 i.e 5 nearest neighbours
        if group == vote:
            correct += 1

        total += 1


print('Accuracy:', str((correct*100)/total))
