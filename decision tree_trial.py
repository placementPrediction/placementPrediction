# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:08:51 2020

@author: Dr_programmer
"""


# Run this, provided you have installed 
# the required libraries. 

 

# Importing the required packages
import pandas as pd 
import joblib
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


#Global Variables
clf_gini=''
clf_entropy=''


# Function to import Dataset 
def importdata(): 
    balance_data = pd.read_csv('finalplacementdata3.csv') 
    
    # Printing the dataset shape 
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
    
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data

 

# Function to split the dataset 
def splitdataset(balance_data): 
  
    # Separating the target variable 
    #X = balance_data.values[:, 'Quantitative ability','LogicalReasoning','English Competency','Programming','CGPA','Computer Fundamentals','CloudComp','WebServices','DataAnalytics','QualityAssurance','AI']
    X = balance_data.values[:, 1:12]    ## Change this to words/feature columns after adding extra columns...
    Y = balance_data.Placed
    print(X)
 
    # Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100) 
    
    return X, Y, X_train, X_test, y_train, y_test 
    
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
    global clf_gini
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    return clf_gini 
    
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
    global clf_entropy
    clf_entropy = DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 5, min_samples_leaf = 1)
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 


# Function to make predictions 
def prediction(X_test, clf_object): 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
    
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
    
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
    
    print("Report : ", 
    classification_report(y_test, y_pred)) 

 

# Driver code 
def main():
  
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
    
    # Prediction Phase
    # Prediction using gini 
    print("Results Using Gini Index:")
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
    
    # Prediction using entropy 
    print("Results Using Entropy:") 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    #Creating the model file
    joblib.dump(clf_gini, 'clf_GiniModel.model')
    
# Calling main function 
if __name__=="__main__": 
    main() 

 

