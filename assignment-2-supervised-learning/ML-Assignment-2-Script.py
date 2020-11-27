# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:24:59 2020

@author: Gearóid Sheehan (R00151523)
"""

#Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection
from PIL import Image
import time
import math
            
"""
FUNCTIONS FOR EACH TASK
"""

#Task 1 - Pre-processing and visualisation
def pprocess_and_visualize(shoes_df):
    
    #Checking for nan values
    print(shoes_df.isnull().values.any())
    
    #Seperate labels from the rest of the dataframe
    labels = shoes_df['label']
    
    #Count number of each type of shoe in the dataframe
    numSneakers = len(labels.loc[labels == 0]) 
    numAnkleBoots = len(labels.loc[labels == 1])
    
    #Print count of each type of shoe in dataframe
    print('Number of sneakers in the dataset:', numSneakers)
    print('Number of ankle boots in the dataset:', numAnkleBoots)
    
    def plot_img(index):
        i = index # set any valid index of an image
        label_1 = shoes_df.values[i][0] # retrieve label from first colum in dataframe
        im_buf = shoes_df.values[i][1:] # create flat array of only the pixels of the given image 
        axis_len = int(math.sqrt(im_buf.shape[0])) # calculate the dimensions of the square image
        im_array = np.int8(np.reshape(im_buf, (axis_len, axis_len))) # create a 2D array from flat array
        img = Image.fromarray(im_array, 'L') # convert to a PIL.Image object ('L' is for grayscale)
        print(f'Label: {label_1}')
        plt.imshow(np.asarray(img))
    
    #NB!!!!!!!!!!! Later implement random allocation of a row with 1/0
    
    #Plot image for sneaker class
    plot_img(2)
    
    #Plot image for ankle boot class
    plot_img(6)
    
    return shoes_df
    
#Task 2 - Perceptron
def perceptron(shoes_df):
    
    target = shoes_df['label']
    data = shoes_df.drop(columns=['label'])

    #Lists to hold results from confusion matrix
    true_pos = []        
    true_neg = []        
    false_pos = []        
    false_neg = []
    
    kf = model_selection.KFold(n_splits=3, shuffle=True)
    
    for train_index, test_index in kf.split(shoes_df):
        
        train_data = data.iloc[train_index]
        train_target = target.iloc[train_index]
        test_data = data.iloc[test_index]
        test_target = target.iloc[test_index]
        
        #List for predicted labels from the data source
        predicted_labels = []
        
        #Set the classifier to the Perceptron model
        clf = linear_model.Perceptron()

        #Train the classifier and measure the processing time
        start_time_train = time.time()
        clf.fit(train_data, train_target)
        end_time_train = time.time()
        elapsed_time_train = end_time_train - start_time_train
        
        #Save the list of predicted labels in a list and measure predicting time
        start_time_predict = time.time()
        predicted_labels = clf.predict(train_data)
        end_time_predict = time.time()
        elapsed_time_predict = end_time_predict - start_time_predict
     
        print("Time taken to train:", elapsed_time_train)
        print("Time taken to predict:", elapsed_time_predict)
        
        score = metrics.accuracy_score(train_target, predicted_labels)
        
        print("\nPerceptron accuracy score: ", score)
        
        #Run target data and predicted labels list through confusion matrix and append results
        C = metrics.confusion_matrix(target[train_index], predicted_labels)
        
        true_pos.append(C[0,0])            
        true_neg.append(C[1,1])                        
        false_pos.append(C[1,0])            
        false_neg.append(C[0,1])
        
        #Print Confusion Matrix
        print("\nConfusion Matrix:")
        print(C)
        
        #Print Confusion Matrix
        print()
        print("True Sneakers:", np.sum(true_pos))
        print("False Sneakers:", np.sum(false_neg))
        print("False Ankle Boots:", np.sum(false_pos))
        print("True Ankle Boots:", np.sum(true_neg))
        
    print(predicted_labels)
        
#Task 3 - Support Vector Machine
def support_vector_machine(shoes_df):
    
    target = shoes_df['label']
    data = shoes_df.drop(columns=['label'])
    
    #Lists to hold results from confusion matrix
    true_pos = []        
    true_neg = []        
    false_pos = []        
    false_neg = []
    
    kf = model_selection.KFold(n_splits=3, shuffle=True)
    
    for train_index, test_index in kf.split(shoes_df):
        
        train_data = data.iloc[train_index]
        train_target = target.iloc[train_index]
        test_data = data.iloc[test_index]
        test_target = target.iloc[test_index]
        
        clf_rbf = svm.SVC(kernel="rbf", gamma=1e-3)    
        clf_linear = svm.SVC(kernel="linear", gamma=1e-4)    
    
        clf_rbf.fit(train_data[train_index], train_target[train_index])
        prediction_rbf = clf_rbf.predict(train_data[test_index])
    
        clf_linear.fit(train_data[train_index], train_target[train_index])
        prediction_linear = clf_linear.predict(train_data[test_index])
        
        score_rbf = metrics.accuracy_score(train_target[test_index], prediction_rbf)
        score_linear = metrics.accuracy_score(train_target[test_index], prediction_linear)
        
        print("SVM with RBF kernel accuracy score: ", score_rbf)
        print("SVM with Sigmoid kernel accuracy score: ", score_linear)
        print()
        
#Task 4 - Comparison
def comparison():
    print()
    
"""
DRIVER CODE AND MAIN FOR EACH TASK
"""

#Main function
def main():
    
    #Read in excel file to pandas dataframe
    shoes = pd.read_csv('product_images.csv')
    
    #TASK 1 DRIVER
    shoes_df = pprocess_and_visualize(shoes)
    
    pr
    #TASK 2 DRIVER
    perceptron(shoes_df)
    
    #TASK 3 DRIVER
    #support_vector_machine(shoes_df)
    
    #TASK 4 DRIVER
    comparison()

#MAIN DRIVER
main()   