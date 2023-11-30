#IMPORTS -------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


#CREATING DATAFRAMES -------------------------------------------------------------------------------------------------------------
#Converting All Text Files To Dataframes (Training Data, Test Data, And Training Label Data)
def transformToDataframes(TextData):
    dataset = pd.read_csv(TextData, sep='\t', header=None, engine='python')
    #Mark Missing Values With Null Placeholder
    dataset.replace(1e+99, np.nan, inplace=True)
    print(TextData[15:], ": ", dataset.shape)
    return dataset

#DATASET ONE - Training, Testing, Labels
TrainingData1 = transformToDataframes('./Project Data/TrainData1.txt')
TestData1 = transformToDataframes('./Project Data/TestData1.txt')
TrainingLabel1 = transformToDataframes('./Project Data/TrainLabel1.txt')
print()
#DATASET FOUR - Training, Testing, Labels
TrainingData4 = transformToDataframes('./Project Data/TrainData4.txt')
TestData4 = transformToDataframes('./Project Data/TestData4.txt')
TrainingLabel4 = transformToDataframes('./Project Data/TrainLabel4.txt')
print()
#DATASET FIVE - Training, Testing, Labels
TrainingData5 = transformToDataframes('./Project Data/TrainData4.txt')
TestData5 = transformToDataframes('./Project Data/TestData4.txt')
TrainingLabel5 = transformToDataframes('./Project Data/TrainLabel4.txt')




#MISSING VALUE ESTIMATION ---------------------------------------------------------------------------------------------------------
# Impute the entire dataset using the best n neighbor found previously (best k = 1, euclidean distance)
knnImpute = KNNImputer(n_neighbors=1, metric='nan_euclidean', add_indicator=False)
TrainingData1 = pd.DataFrame(knnImpute.fit_transform(TrainingData1), columns = TrainingData1.columns)




#CLASSIFIER EVALUATION --------------------------------------------------------------------------------------------------
#Applying The Class Training Labels To Datasets
TrainingData1['class'] = TrainingLabel1

#KNN CLASSIFIER EVALUATION
y = TrainingData1['class']
X = TrainingData1.drop('class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
K = [] 
training = [] 
test = [] 
scores = {} 
  
for k in range(1, 21): 
    knn_clf = KNeighborsClassifier(n_neighbors = k) 
    knn_clf.fit(X_train, y_train) 

    y_pred = knn_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    print('K=', k, ', Accuracy: ', accuracy, ', Precision: ', precision, ', Recall: ', recall)
    print()


'''
def applyKNNClassifier(TrainData, TrainLabel):
    #Applying The Class Training Labels To Datasets
    TrainData['class'] = TrainLabel

    #KNN CLASSIFIER EVALUATION
    y = TrainData['class']
    X = TrainData.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    bestKScore = (0, 0, 0, 0)
    for k in range(1, 21): 
        knn_clf = KNeighborsClassifier(n_neighbors = k) 
        knn_clf.fit(X_train, y_train) 

        y_pred = knn_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')

        if accuracy > bestKScore[1]:
            bestKScore = (k, accuracy, precision, recall)
        print('K = ', k, ', Accuracy: ', accuracy, '\n')

    print('K = ', bestKScore[0])
    print('Accuracy: ', bestKScore[1])
    print('Precision: ', bestKScore[2])
    print('Recall: ', bestKScore[3])
    print()

applyKNNClassifier(TrainingData1, TrainingLabel1)
'''