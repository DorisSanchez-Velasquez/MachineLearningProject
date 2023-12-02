#IMPORTS -------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats import skew

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


#CREATING DATAFRAMES -------------------------------------------------------------------------------------------------------------
#Converting All Text Files To Dataframes (Training Data, Test Data, And Training Label Data)
def transformToDataframes(TextData):
    dataset = pd.read_csv(TextData, sep='\t', header=None, engine='python')
    #Mark Missing Values With Null Placeholder
    dataset.replace(1e+99, np.nan, inplace=True)
    return dataset

#Fixing the shaping error for dataframes with different separators where Datasets 4 and 5 need to be split
def fixingDataframeShapeError(dataset):
    dataset = dataset[0].str.split(r'\s+', expand=True)
    dataset = dataset.drop([0], axis=1)
    #Changing all columns from type object to type float
    for column in dataset.columns:
        dataset[column] = dataset[column].astype(float)
    return dataset


#DATASET ONE - Training, Testing, Labels
TrainingData1 = transformToDataframes('./Project Data/TrainData1.txt')
TestData1 = transformToDataframes('./Project Data/TestData1.txt')
TrainingLabel1 = transformToDataframes('./Project Data/TrainLabel1.txt')
print("TrainingData1: ", TrainingData1.shape, '\n')

#DATASET FOUR - Training, Testing, Labels
TrainingData4 = fixingDataframeShapeError(transformToDataframes('./Project Data/TrainData4.txt'))
TestData4 = transformToDataframes('./Project Data/TestData4.txt')
TrainingLabel4 = transformToDataframes('./Project Data/TrainLabel4.txt')
print("TrainingData4: ", TrainingData4.shape, '\n')

#DATASET FIVE - Training, Testing, Labels
TrainingData5 = fixingDataframeShapeError(transformToDataframes('./Project Data/TrainData5.txt'))
TestData5 = transformToDataframes('./Project Data/TestData5.txt')
TrainingLabel5 = transformToDataframes('./Project Data/TrainLabel5.txt')
print("TrainingData5: ", TrainingData5.shape, '\n')




#MISSING VALUE ESTIMATION ---------------------------------------------------------------------------------------------------------
# Impute the entire dataset using the best n neighbor found previously (best k = 1, euclidean distance)
knnImpute = KNNImputer(n_neighbors=1, metric='nan_euclidean', add_indicator=False)
TrainingData1 = pd.DataFrame(knnImpute.fit_transform(TrainingData1), columns = TrainingData1.columns)
TrainingData4 = pd.DataFrame(knnImpute.fit_transform(TrainingData4), columns = TrainingData4.columns)
TrainingData5 = pd.DataFrame(knnImpute.fit_transform(TrainingData5), columns = TrainingData5.columns)


def applyMeanMedianImputation(dataset, skew_threshold=0.5):
    for column in dataset.columns:
        column_skewness = skew(dataset[column].dropna())

        if abs(column_skewness) < skew_threshold:
            dataset[column].fillna(dataset[column].mean(), inplace=True)
        else:
            dataset[column].fillna(dataset[column].median(), inplace=True)
    return dataset

#TrainingData4 = applyMeanMedianImputation(TrainingData4)
#TrainingData5 = applyMeanMedianImputation(TrainingData5)


#NORMALIZATION ---------------------------------------------------------------------------------------------------------
#Applying logarithmic normalization to dataset to fix skewed data distributions
def applyLogNormalize(dataset):
    normalizedData = dataset.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()).round(2))
    return normalizedData





#CLASSIFIER EVALUATION --------------------------------------------------------------------------------------------------

#Applying The K Nearest Neighbor Classification Model
def applyKNNClassifier(TrainData, TrainLabel):
    #Applying The Class Training Labels To Datasets
    TrainData['class'] = TrainLabel

    #KNN CLASSIFIER EVALUATION
    targetClass = TrainData['class']
    data = TrainData.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, targetClass, test_size=0.2)
    
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

    print('KNN Classifier:')
    print('K = ', bestKScore[0])
    print('Accuracy: ', bestKScore[1])
    print('Precision: ', bestKScore[2])
    print('Recall: ', bestKScore[3], '\n')




#Support Vector Machine Classifier
def applySVMClassifier(TrainData, TrainLabel):
    #Normalize The Data Prior To Classifier Training
    TrainData = applyLogNormalize(TrainData)

    #Applying The Class Training Labels To Datasets
    TrainData['class'] = TrainLabel

    #SVM CLASSIFIER EVALUATION
    targetClass = TrainData['class']
    data = TrainData.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, targetClass, test_size = 0.30, random_state=42)
    
    #Apply PCA To Training set
    pca = PCA(n_components = 4)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    SVM_clf = SVC(kernel = 'linear', random_state=42)
    SVM_clf.fit(X_train, y_train)

    y_pred = SVM_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')

    print("SVM Classifier: ")
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall, '\n')




#Logistic Regression Classifier
def applyLogisticRegression(TrainData, TrainLabel):
    #Normalize The Data Prior To Classifier Training
    TrainData = applyLogNormalize(TrainData)

    #Applying The Class Training Labels To Datasets
    TrainData['class'] = TrainLabel

    #LOG REGRESSION CLASSIFIER EVALUATION
    targetClass = TrainData['class']
    data = TrainData.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, targetClass, test_size = 0.30, random_state=42)
    
    logistic_clf = LogisticRegression(random_state=42)
    logistic_clf.fit(X_train, y_train)

    y_pred = logistic_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')

    print("Logistic Regression Classifier: ")
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall, '\n')





#RANDOM FOREST CLASSIFIER
def applyRandomForestClassifier(TrainData, TrainLabel):
    #Normalize The Data Prior To Classifier Training
    TrainData = applyLogNormalize(TrainData)

    #Applying The Class Training Labels To Datasets
    TrainData['class'] = TrainLabel

    #LOG REGRESSION CLASSIFIER EVALUATION
    targetClass = TrainData['class']
    data = TrainData.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, targetClass, test_size = 0.30, random_state=42)
    
    #Apply PCA To Training set
    pca = PCA(n_components = 4)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    forest_clf = RandomForestClassifier(n_estimators = 25)
    forest_clf.fit(X_train, y_train)

    y_pred = forest_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')

    print("Random Forest Classifier: ")
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall, '\n')


#CLASSIFIER APPLICATION --------------------------------------------------------------------------------------------------

print('TRAINING DATA ONE', '-'*40)
applyKNNClassifier(TrainingData1, TrainingLabel1)
applySVMClassifier(TrainingData1, TrainingLabel1)
applyLogisticRegression(TrainingData1, TrainingLabel1)
applyRandomForestClassifier(TrainingData1, TrainingLabel1)

print('TRAINING DATA FOUR', '-'*40)
applyKNNClassifier(TrainingData4, TrainingLabel4)
applySVMClassifier(TrainingData4, TrainingLabel4)
applyLogisticRegression(TrainingData4, TrainingLabel4)
applyRandomForestClassifier(TrainingData4, TrainingLabel4)

print('TRAINING DATA FIVE', '-'*40)
applyKNNClassifier(TrainingData5, TrainingLabel5)
applySVMClassifier(TrainingData5, TrainingLabel5)
applyLogisticRegression(TrainingData5, TrainingLabel5)
applyRandomForestClassifier(TrainingData5, TrainingLabel5)
