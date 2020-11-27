# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:08:13 2020

@author: Gearóid Sheehan (R00151523)
"""

#Library Imports
import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import model_selection

"""
FUNCTIONS FOR EACH TASK
"""

#Task 1 - Function to split the excel into lists and print sizes. Returns tuple of lists
def split_and_count(df):
    
    #Models to be passed into inner_func
    train = 'train'
    test = 'test'
    
    #Inner function which splits the dataframe df passed in into two series, counts the 
    #positive and negative reviews in each model passed to it and then prints the count to screen.
    def inner_func(model, df):
        
        #Splitting data for model passed in
        df = df[df['Split'] == model]
        
        #Placing each required series from dataframe in variable
        data_list = df['Review']
        labels_list = df['Sentiment']
        
        #Variables to store lengths of models lists. Global so can be accessed from other functions
        global labels_positive_train
        global labels_negative_train
        global labels_positive_test
        global labels_negative_test
        
        #Printing size of lists and setting sizes of above global variables depending on model passed in
        if model == 'train':
            labels_positive_train = len(labels_list.loc[labels_list == 'positive'])
            labels_negative_train = len(labels_list.loc[labels_list == 'negative'])
            print('Number of positive reviews in the', model,' set:', labels_positive_train)
            print('Number of negative reviews in the', model,' set:', labels_negative_train)
            
        elif model == 'test':
            labels_positive_test = len(labels_list.loc[labels_list == 'positive'])
            labels_negative_test = len(labels_list.loc[labels_list == 'negative'])
            print('Number of positive reviews in the', model,' set:', labels_positive_test)
            print('Number of negative reviews in the', model,' set:', labels_negative_test)
        
        #Returns the two series, which are each models Review and Sentiment columns, as lists
        return (data_list.values.tolist(), labels_list.values.tolist())

    #Calling inner function for each model and setting each returned list from inner_func to a new variable
    training_result = inner_func(train, df)
    training_list_data = training_result[0]
    training_list_labels = training_result[1]
    
    test_result = inner_func(test, df)
    test_list_data = test_result[0]
    test_list_labels = test_result[1]
    
    #Returning the four lists required for Task 1
    return (training_list_data, training_list_labels, test_list_data, test_list_labels)
    
#Task 2 - Function to clean data, count word occurences in training set and get minimum word length/occurance
def extract_relevant_features(training_list_data, minimum_word_length, minimum_word_occurance):
    
    #Converts training list to pandas series
    training_list_data = pd.Series(training_list_data)
    
    #Using a regular expression which removes characters which are not alphanumeric and creates dataframe
    cleaned_training_list_data = training_list_data.to_frame().replace('[^a-zA-Z0-9 ]', '', regex=True)
    cleaned_training_list_data.columns = ['Review']
    
    #Checks if data has been cleaned - Commented out after use
    #characters = Counter(''.join(cleaned_training_list_data.unstack().values))
    #print(characters)

    #Extracting individual words from the dataframe and converting to lower
    individual_words = cleaned_training_list_data['Review'].str.lower().str.split(expand=True).stack().value_counts().to_frame()
    
    individual_words['Word'] = individual_words.index
    individual_words.reset_index(inplace=True, drop=True)
    individual_words.rename(columns={0 :'Count'}, inplace=True)
    
    #Removes words occurring less than the minimum word occurance
    indexNames = individual_words[ individual_words['Count'] < minimum_word_occurance].index
    individual_words.drop(indexNames , inplace=True)
    
    #Removes words smaller than the input minimum word length
    indexNames = individual_words[ individual_words['Word'].str.len() < minimum_word_length].index
    individual_words.drop(indexNames , inplace=True)
    
    #Checks the count of each individual word in the full training set - Commented out after use
    print("\nCOUNT OF INDIVIDUAL WORDS IN FULL TRAINING SET")
    print(individual_words)

    #Return dataframe as list
    global word_list
    word_list = individual_words['Word'].tolist()
    
    return word_list

#Task 3 - Function to to extract the set of all words of a minimum length and with a minimum number of occurrences from the reviews in the training set
def count_feature_frequencies(training_list_data, training_list_labels, word_list, minimum_word_length, minimum_word_occurance):
    
    #Convert lists to dataframe
    training_list_df = pd.DataFrame(training_list_data, columns=['Review'])
    training_list_df['Sentiment'] = training_list_labels
    
    #Extracts positive and negative reviews from dataframe and stores
    positive_reviews = training_list_df.loc[training_list_df['Sentiment'] == 'positive']
    negative_reviews = training_list_df.loc[training_list_df['Sentiment'] == 'negative']
    
    #Function to count the rows where each word occurs at least once
    def count_rows_words_appear(dct, df_reviews, word_list, name):
        
        #Loops over each word returns a count of rows where words occurs at least once
        for word in word_list:
            dct[word] = len(df_reviews[df_reviews['Review'].str.contains(word)])                                                                                
        
        #Checks how many reviews each word occured in at least once - Commented out after use
        print("\nCOUNT OF " + name + " REVIEWS CONTAINING WORDS WHICH ARE IN THE TRAINING SET\n" +
              "AND ARE " + str(minimum_word_length) + " CHARACTERS LONG AND OCCUR AT LEAST " + 
              str(minimum_word_occurance) + " TIMES\n")
        
        print(dct)

        return dct
    
    #Creating dictionaries for both positive and negative reviews and populate them with return from above function
    positive_dct = dict()
    positive_dct = count_rows_words_appear(positive_dct, positive_reviews, word_list, 'POSITIVE')
    
    negative_dct = dict()
    negative_dct = count_rows_words_appear(negative_dct, negative_reviews, word_list, 'NEGATIVE')
    
    #Return populated dictionaries
    return positive_dct, negative_dct
    
#Task 4 - Function to calculate the feature likelihoodds and priors
def calc_likelihoods_priors(labels_pos_train, labels_neg_train, pos_dct, neg_dct):
    
    def calc_likelihood(likelihood_dct, label_length, dct):
        
        #Loop over each key and value in dictionary
        for k,v in dct.items():
             
            #Calculates P value with Laplace smoothing of 1
            likelihood_dct[k] = (v + 1) / (sum(dct.values()) + len(dct))
                                                                                
        return likelihood_dct
    
    #Creates dictionaries and populates them with each word and its calculated P value
    pos_likelihood_dct = dict()
    pos_likelihood_dct = calc_likelihood(pos_likelihood_dct, labels_pos_train, pos_dct)
    
    neg_likelihood_dct = dict()
    neg_likelihood_dct = calc_likelihood(neg_likelihood_dct, labels_neg_train, neg_dct)
 
    #Checks the dictionaries for the words and their P values - Commented out after use
    
    #print(pos_likelihood_dct)
    #print(neg_likelihood_dct)
     
    #print(sum(pos_likelihood_dct.values()))
    #print(sum(neg_likelihood_dct.values()))
    
    #Calculates Priors and prints to console
    print('\nPRIORS')
    
    pos_prior = labels_pos_train / (labels_pos_train + labels_neg_train)
    print('Prior for positive reviews: ' + str(pos_prior))
    
    neg_prior = labels_neg_train / (labels_pos_train + labels_neg_train)
    print('Prior for negative reviews: ' + str(neg_prior))
    
    return pos_likelihood_dct, neg_likelihood_dct, pos_prior, neg_prior

#Task 5 - Function to calculate P values of each word being present in review for both positive and negative, Laplace smoothing also of 1
def max_likelihood_classification(pos_likelihood_dct, neg_likelihood_dct, pos_prior, neg_prior, review):
    
    #Split passed in review into individual words and create dictionaries
    words_in_review = review.split()
    pos_dct = dict()
    neg_dct = dict()
    
    #Naive Bayes Classifier Algorithm
    def bayes_classifier(likelihood_dct, dct, prior):
        
        #Checks if reviews words are in dictionary, if they are sets their P value to previously calculated, if not sets to 0
        for word in words_in_review:
            if word in likelihood_dct:
                dct[word] = likelihood_dct[word]
            
            else:
                dct[word] = 0
        
        #Bayes Theorem Implementation
        for k, v in dct.items():
         
            dct[k] = (v + 1) * prior / (sum(dct.values()) + len(dct))
                 
        total = 0                      
        
        #Using logarithms to ensure the results are numerically stable
        for i, x in dct.items():
            
            dct[i] = np.log(x)
            total = total + dct[i]

        return math.exp(total)
    
    #Calls above function to get both positive and negative outcome data
    pos_outcome = bayes_classifier(pos_likelihood_dct, pos_dct, pos_prior)
    neg_outcome = bayes_classifier(neg_likelihood_dct, neg_dct, neg_prior)
    
    #Prints outcomes to console - Commented out after use
    #print(pos_outcome)
    #print(neg_outcome)
    
    global outcome
    
    #Decides whether results are positive or negative and returns binary and text results
    if pos_outcome > neg_outcome or pos_outcome == neg_outcome:
        outcome = 0
        val = 'Positive'
        
    elif pos_outcome < neg_outcome:
        outcome = 1
        val = 'Negative'
     
    return outcome, val
        
#Task 6 - Function which uses the classifier created in previous functions and trains model, creates confusion matrix and tests accuracy using KFold Cross Validation 
def evalutation_of_results(training_data, training_labels, minimum_word_occurance):
    
    #Creates dataframe out of training data and labels, setting the sentiment values to a binary 1 or 0 
    df = pd.DataFrame(training_data, columns=['Review'])
    df['Sentiment'] = training_labels
    
    df['Sentiment'] = df['Sentiment'].map({"positive":0,"negative":1})
    
    data = df[["Review","Sentiment"]]    
    target = df["Sentiment"]
    
    #Creates cross-validation object which returns stratified folds of 3 splits
    kf = model_selection.StratifiedKFold(n_splits=3)
    
    #List to store accuracy values for the mean calculation
    accuracy_avg_train = []
    accuracy_avg_test = []
    
    #List to store accuracy value from each word length
    accuracy_list = []
    
    for k in range(1, 11):       
                
        #Lists to hold results from confusion matrix
        true_pos = []        
        true_neg = []        
        false_pos = []        
        false_neg = []
    
        #Reload tasks 2, 3 and 4 in order to test accuracy for different word lengths 1 - 10, or k
        word_list = extract_relevant_features(training_data, k, minimum_word_occurance)
        dictionaries = count_feature_frequencies(training_data, training_labels, word_list,  k, minimum_word_occurance)
        likelihoods_priors = calc_likelihoods_priors(labels_positive_train, labels_negative_train, dictionaries[0], dictionaries[1])
        
        #Loop through each train and/or test index for each split 
        for train_index, test_index in kf.split(data, target):
        
            #List for predicted labels from the data source
            predicted_labels = []
            
            #Loop through test index and add each sentiment result from classifier to predicted labels list
            for x in train_index:
                outcome = max_likelihood_classification(likelihoods_priors[0], likelihoods_priors[1], likelihoods_priors[2], likelihoods_priors[3], str(df['Review'].values[x]))
                predicted_labels.append(outcome[0])
             
            #Run target data and predicted labels list through confusion matrix and append results
            C = metrics.confusion_matrix(target[train_index], predicted_labels)
            
            true_pos.append(C[0,0])            
            true_neg.append(C[1,1])                        
            false_pos.append(C[1,0])            
            false_neg.append(C[0,1])
       
            accuracy_avg_train.append(accuracy_score(target[train_index], predicted_labels))
            #Print Accuracy Score to console
            accuracy = accuracy_score(target[train_index], predicted_labels)
            print('Accuracy Score for word length ' + str(k) + ':', accuracy)
            accuracy_list.append(accuracy)
        
        #Calculate mean accuracy for train
        mean_accuracy_train = sum(accuracy_avg_train) / len(accuracy_avg_train)
        print('Mean accuracy for train sample: ', mean_accuracy_train)
        
        #Print to console each result count as a percentage
        print('\nRESULTS')
        print('True positive:', np.sum(true_pos) / len(df.index) * (100 / 1), '%')        
        print('True negative:', np.sum(true_neg) / len(df.index) * (100 / 1), '%')        
        print('False positive:', np.sum(false_pos) / len(df.index) * (100 / 1), '%')        
        print('False negative:', np.sum(false_neg) / len(df.index) * (100 / 1), '%')        
        print()        
     
    
        #Retrieve Optimal Word Length
        highest_accuracy = max(accuracy_list)
        optimal_word_length = (accuracy_list.index(highest_accuracy) + 1)
        print('\nOptimal Word Length: ' + str(optimal_word_length))
        
        #Run KFold on Test List with Optimal word length
        word_list = extract_relevant_features(training_data, k, minimum_word_occurance)
        dictionaries = count_feature_frequencies(training_data, training_labels, word_list, optimal_word_length, minimum_word_occurance)
        likelihoods_priors = calc_likelihoods_priors(labels_positive_train, labels_negative_train, dictionaries[0], dictionaries[1])
        
        #Loop through each train and/or test index for each split 
        for train_index, test_index in kf.split(data, target):
        
            #List for predicted labels from the data source
            predicted_labels = []
            
            #Loop through test index and add each sentiment result from classifier to predicted labels list
            for x in test_index:
                outcome = max_likelihood_classification(likelihoods_priors[0], likelihoods_priors[1], likelihoods_priors[2], likelihoods_priors[3], str(df['Review'].values[x]))
                predicted_labels.append(outcome[0])
             
            #Run target data and predicted labels list through confusion matrix and append results
            C = metrics.confusion_matrix(target[test_index], predicted_labels)
            true_pos.append(C[0,0])            
            true_neg.append(C[1,1])                        
            false_pos.append(C[1,0])            
            false_neg.append(C[0,1]) 
            
            accuracy_avg_test.append(accuracy_score(target[train_index], predicted_labels))
            #Print Accuracy Score to console
            accuracy = accuracy_score(target[train_index], predicted_labels)
            print('Accuracy Score for word length ' + str(k) + ':', accuracy)
            accuracy_list.append(accuracy)
            
        mean_accuracy_test = sum(accuracy_avg_test) / len(accuracy_avg_test)
        print('Mean accuracy for test sample: ', mean_accuracy_test)
        
        #Print to console each result count as a percentage
        print('\nRESULTS')
        print('True positive:', (np.sum(true_pos) / len(df.index)) * (100 / 1), '%')        
        print('True negative:', np.sum(true_neg) / len(df.index) * (100 / 1), '%')        
        print('False positive:', np.sum(false_pos) / len(df.index) * (100 / 1), '%')        
        print('False negative:', np.sum(false_neg) / len(df.index) * (100 / 1), '%')        
        print()        
    
"""
DRIVER CODE AND MAIN FOR EACH TASK
"""

#Main function
def main():
  
    #Read in excel file to pandas dataframe    
    df = pd.read_excel(r'movie_reviews.xlsx')
    
    #TASK 1 DRIVER
    print('\n-----TASK 1-----') 
    
    #Run function 1 and store tuple of lists returned in t1_lists variable
    t1_lists = split_and_count(df)
    
    #TASK 2 DRIVER
    print('\n-----TASK 2-----')
    
    #User input for minimum word length/occurance
    print('Please enter minimum word length:')
    minimum_word_length = int(input())
    print('Please enter minimum word occurance')
    minimum_word_occurance = int(input())
    
    #Run function 2 and store list of words extracted in word_list variable 
    word_list = extract_relevant_features(t1_lists[0], minimum_word_length, minimum_word_occurance)
    
    #TASK 3 DRIVER
    print('\n-----TASK 3-----')
    
    #Run function 3 and store dictionaries returned in dictionaries variable
    dictionaries = count_feature_frequencies(t1_lists[0], t1_lists[1], word_list,  minimum_word_length, minimum_word_occurance)
    
    #TASK 4 DRIVER
    print('\n-----TASK 4-----')
    
    #Run function 4 and return the likelihoods and priors calculated in likelihoods_priors variable
    likelihoods_priors = calc_likelihoods_priors(labels_positive_train, labels_negative_train, dictionaries[0], dictionaries[1])
    
    #TASK 5 DRIVER
    print('\n-----TASK 5-----')
    
    #New external review to test in function 5
    review = """With the current global situation seemingly beyond satire and the planet tilting on an axis
                    of surreal dread you may wonder what Sacha Baron Cohen hopes to achieve by bringing his most
                    famous creation back to punk and prank America in 2020. As with the recent return of Spitting 
                    Image, surely the grotesques who currently sit in power and the cracked theories of their 
                    followers are immune to Cohen’s mix of satire, slapstick, and the comedy of cringe."""
    
    #Run function 5 and store prediction returned in prediction variable
    prediction = max_likelihood_classification(likelihoods_priors[0], likelihoods_priors[1], likelihoods_priors[2], likelihoods_priors[3], review)
    print('Prediction: ', prediction[1])
    
    #TASK 6 DRIVER
    print('\n-----TASK 6-----')
    evalutation_of_results(t1_lists[0], t1_lists[1], minimum_word_occurance)

#MAIN DRIVER
main()