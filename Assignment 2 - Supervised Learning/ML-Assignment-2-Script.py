# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:24:59 2020

@author: Gearóid Sheehan (R00151523)
"""

#Library Imports
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection
from PIL import Image
import timeit
import math
            
"""
FUNCTIONS FOR EACH TASK
"""

#Task 1 - Pre-processing and visualisation
def pprocess_and_visualize(shoes_df):
    
    #Checking for nan values
    print("\nIs there any nan values in the dataset?:", shoes_df.isnull().values.any())
    
    #Seperate labels from the rest of the dataframe
    labels = shoes_df["label"]
    
    #Count number of each type of shoe in the dataframe
    numSneakers = len(labels.loc[labels == 0]) 
    numAnkleBoots = len(labels.loc[labels == 1])
    
    #Print count of each type of shoe in dataframe
    print("\nNumber of sneakers in the dataset:", numSneakers)
    print("Number of ankle boots in the dataset:", numAnkleBoots)
    
    #Prepare image for plotting
    def plot_img(index) -> None:
        
        # set any valid index of an image and create flat array of only the pixels of the given image 
        i = index
        im_buf = shoes_df.values[i][1:]
        #Calculate the dimensions of the square image and create a 2D array from flat array
        axis_len = int(math.sqrt(im_buf.shape[0]))
        im_array = np.int8(np.reshape(im_buf, (axis_len, axis_len)))
        
        #Convert to a PIL.Image object ('L' is for grayscale)
        img = Image.fromarray(im_array, 'L')
        plt.figure()
        plt.imshow(np.asarray(img))
    
    #Get random image index of sneaker
    rand_sneaker = shoes_df.index[shoes_df['label'] == 0].tolist()
    rand_sneaker = random.choice(rand_sneaker)
    
    #Get random image index of boot
    rand_boot = shoes_df.index[shoes_df['label'] == 1].tolist()
    rand_boot = random.choice(rand_boot)
    
    #List to store random indexes
    images = [rand_sneaker, rand_boot]
    
    #Plot 1 image of sneaker and 1 of boot
    for file in images:
        plot_img(file)
    
    return shoes_df
    
#Task 2 - Perceptron
def perceptron(shoes_df, splits):
    
    #Split data into 2 dataframes, one for target(label) and one for data
    target = shoes_df['label']
    data = shoes_df.drop(columns=['label'])

    #Lists to hold results from confusion matrix
    true_pos = []        
    true_neg = []        
    false_pos = []        
    false_neg = []
    
    #Lists to hold time values and accucary values for each split
    train_times = []
    predict_times = []
    predict_acc = []
    
    #Create K-Fold for Stratified K-fold
    kf = model_selection.KFold(n_splits=splits, shuffle=True)
    
    for train_index, test_index in kf.split(data, target):
        
        #Split the data into train and test data
        train_data = data.iloc[train_index]
        train_target = target.iloc[train_index]
        test_data = data.iloc[test_index]
        test_target = target.iloc[test_index]

        #List for predicted labels from the data source
        predicted_labels = []
        
        #Set the classifier to the Perceptron model
        clf = linear_model.Perceptron()

        #Train the classifier and measure the processing time
        start_train = timeit.default_timer()
        clf.fit(train_data, train_target)
        end_train = timeit.default_timer()
        elapsed_time_train = end_train - start_train
        
        #Add the processing time to the list of train times
        train_times.append(elapsed_time_train)
        
        #Save the list of predicted labels in a list and measure predicting time
        start_predict = timeit.default_timer()
        predicted_labels = clf.predict(test_data)
        end_predict = timeit.default_timer()
        elapsed_time_predict = end_predict - start_predict
        
        #Add the processing time to the list of prediction times
        predict_times.append(elapsed_time_predict)
        
        print("\nTime taken to train Perceptron Classifier:", elapsed_time_train)
        print("Time taken to predict Perceptron Classifier:", elapsed_time_predict)
        
        #Get accuracy score for Perceptron classifier
        score = metrics.accuracy_score(test_target, predicted_labels)
        
        #Add the prediciton accuracy to the list of prediction accuracies
        predict_acc.append(score)
        
        print("\nPerceptron accuracy score: ", score)
        
        #Run target data and predicted labels list through confusion matrix and append results
        C = metrics.confusion_matrix(test_target, predicted_labels)
        
        true_pos.append(C[0,0])            
        true_neg.append(C[1,1])                        
        false_pos.append(C[1,0])            
        false_neg.append(C[0,1])
        
        #Print Confusion Matrix
        print("\nConfusion Matrix for Perceptron Classifier:")
        print(C)
        
        print()
        print("True Sneakers:", np.sum(true_pos))
        print("False Sneakers:", np.sum(false_neg))
        print("False Ankle Boots:", np.sum(false_pos))
        print("True Ankle Boots:", np.sum(true_neg))
        
    #Calculate average training time and print to console
    avg_train_time = sum(train_times) / len(train_times)
    
    print()
    print("Minimum time to train a sample:", min(train_times))
    print("Maximum time to train a sample:", max(train_times))
    print("Average time to train a sample:", avg_train_time)
    
    #Calculate average prediciton time and print to console
    avg_prediction_time = sum(predict_times) / len(predict_times)
    
    print()
    print("Average time to predict a sample:", avg_prediction_time)
    
    #Calculate average prediciton accuracy and print to console
    avg_predict_acc = sum(predict_acc) / len(predict_acc)
    
    print()
    print("Average prediction accuracy of a sample:", avg_predict_acc)
    
#Task 3 - Support Vector Machines
def support_vector_machine(shoes_df, splits):
    
    def run_classifier(clf, name):
    
        #Lists to hold results from confusion matrix
        true_pos = []        
        true_neg = []        
        false_pos = []        
        false_neg = []
    
        #Lists to hold times and prediction accuracies for each split
        train_times = []
        predict_times = []
        predict_acc = []
        
        #Create K-Fold for Stratified K-fold
        kf = model_selection.KFold(n_splits=splits, shuffle=True)
        
        for train_index, test_index in kf.split(shoes_df):
            
            #Split the data into train and test data
            train_data = data.iloc[train_index]
            train_target = target.iloc[train_index]
            test_data = data.iloc[test_index]
            test_target = target.iloc[test_index]
            
            #List for predicted labels from the data source
            predicted_labels = []
        
            #Train the classifier and measure the processing time
            start_train = timeit.default_timer()
            clf.fit(train_data, train_target)
            end_train = timeit.default_timer()
            elapsed_time_train = end_train - start_train
            
            #Append time to train to list so max/min/average can be calculated later
            train_times.append(elapsed_time_train)
            
            #Save the list of predicted labels for the classifier in a list and measure predicting time
            start_predict = timeit.default_timer()
            predicted_labels = clf.predict(test_data)
            end_predict = timeit.default_timer()
            elapsed_time_predict = end_predict - start_predict
            
            #Append time to predict to list so max/min/average can be calculated later
            predict_times.append(elapsed_time_predict)
            
            print("\nTime taken to train " + name + " Classifier:", elapsed_time_train)
            print("Time taken to predict " + name + " Classifier:", elapsed_time_predict)
            
            #Get the accuracy score for the classifier
            score = metrics.accuracy_score(test_target, predicted_labels)
            
            #Append prediction accuracy to list so average can be calculated later
            predict_acc.append(score)
            
            print("SVM with " + name + " accuracy score: ", score)
            
            #If the function is being called for RBF append results as needed to calculate optimal gamma
            if (name == "RBF"):
                gamma_results.append(score)
                
            #Run target data and predicted labels list through confusion matrix and append results
            C = metrics.confusion_matrix(test_target, predicted_labels)
            
            true_pos.append(C[0,0])            
            true_neg.append(C[1,1])                        
            false_pos.append(C[1,0])            
            false_neg.append(C[0,1])
            
            #Print Confusion Matrix
            print("\nConfusion Matrix for " + name + " Classifier:")
            print(C)
            
            print()
            print("True Sneakers:", np.sum(true_pos))
            print("False Sneakers:", np.sum(false_neg))
            print("False Ankle Boots:", np.sum(false_pos))
            print("True Ankle Boots:", np.sum(true_neg))
            
        #Calculate average training time and print to console
        avg_train_time = sum(train_times) / len(train_times)
        
        print()
        print("Minimum time to train a sample:", min(train_times))
        print("Maximum time to train a sample:", max(train_times))
        print("Average time to train a sample:", avg_train_time)
        
        #Calculate average prediction time and print to console
        avg_prediction_time = sum(predict_times) / len(predict_times)
        
        print()
        print("Average time to predict a sample:", avg_prediction_time)
        
        #Calculate average prediciton accuracy and print to console
        avg_predict_acc = sum(predict_acc) / len(predict_acc)
        
        print()
        print("Average prediction accuracy of a sample:", avg_predict_acc)
        
        return avg_train_time, avg_prediction_time, avg_predict_acc
        
    #Split data into 2 dataframes, one for target(label) and one for data
    target = shoes_df['label']
    data = shoes_df.drop(columns=['label'])
    
    #Lists of gamma values to test and to store results from using each
    gamma_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    gamma_results = []
    gamma_used = np.array([])
    
    #Loop through each list and run RBF classifier using each gamma value 
    for y in gamma_list:
        clf_rbf = svm.SVC(kernel="rbf", gamma=y)
        run_classifier(clf_rbf, 'RBF')
        sub_arr = [y] * splits
        gamma_used = np.concatenate((gamma_used, sub_arr))
    
    #Run Linear classifier
    clf_linear = svm.SVC(kernel="linear")
    linear_results = run_classifier(clf_linear, 'Linear')
    
    #Find optimal gamma value, the value with highest accuracy score
    optimal_gamma = gamma_results.index(max(gamma_results))
    optimal_gamma = gamma_used[optimal_gamma]

    #Print optimal gamma value
    print("Optimal Result for Gamma in the RBF Classifier is", optimal_gamma)
        
    #Run the RBF Classifier again, this time just using the optimal gamma value
    clf_rbf = svm.SVC(kernel="rbf", gamma=optimal_gamma)
    rbf_results = run_classifier(clf_rbf, 'RBF')
    
    return rbf_results, linear_results, optimal_gamma 
    
#Task 4 - Comparison
def comparison(results):
    
    print()
    print("****** COMPARE END RESULTS ******")
    
    print()
    print("RBF CLASSIFIER USING OPTIMAL GAMMA OF " + str(results[2]) + ":")
    print("Average time to train RBF Classifier:", results[0][0])
    print("Average time to predict RBF Classifier:", results[0][1])
    print("Average accuracy score of RBF Classifier:", results[0][2])
    
    print()
    print("LINEAR CLASSIFIER:")
    print("Average time to train Linear Classifier:", results[1][0])
    print("Average time to predict Linear Classifier:", results[1][1])
    print("Average accuracy score of Linear Classifier:", results[1][2])
    
    print()
    print("Algorithm chosen: Linear Classifier")
    
    
"""
DRIVER CODE AND MAIN FOR EACH TASK
"""

#Main function
def main():
    
    #Read in excel file to pandas dataframe
    shoes = pd.read_csv("product_images.csv")
    
    percentage = 0
    splits = 0
    
    #Parameterizing the number of samples to use through user input as percentage
    
    #Checks if valid percentage value is entered
    while percentage > 100 or percentage == 0:
        
        #Percentage Input for parameterization
        print("Please enter the percentage of the sample data to use:")
        percentage = int(input()) / 100
    
    shoes = shoes[0:int(percentage*len(shoes))]

    #Take input so user can decide how many splits to use in k-fold cross validation
    
    while splits == 0 or isinstance(splits, int) == False:
    
        #Integer input for number of splits to be used
        print("Please enter the amount of splits to be used:")
        splits = int(input())    

    #TASK 1 DRIVER
    shoes_df = pprocess_and_visualize(shoes)

    #TASK 2 DRIVER
    perceptron(shoes_df, splits)
    
    #TASK 3 DRIVER
    results = support_vector_machine(shoes_df, splits)
    
    #TASK 4 DRIVER
    comparison(results)

#MAIN DRIVER
main()   