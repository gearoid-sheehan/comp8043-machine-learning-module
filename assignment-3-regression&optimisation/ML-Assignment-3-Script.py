# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:33:28 2020

@author: Gearóid Sheehan (R00151523)
"""

#Library Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

"""
FUNCTIONS FOR EACH TASK
"""

#Task 1 - Pre-Processing
def pre_processing(df):
    
    #Checking for nan values
    print("\nIs there any nan values in the dataset?:", df.isnull().values.any())
    print()
    
    #Extract the relevant subsets and print to console
    cut_qualities = df.cut.unique()
    print("Unique cut values in the dataframe:", cut_qualities)
    print()
    
    color_qualities = df.color.unique()
    print("Unique color values in the dataframe:", color_qualities)
    print()
    
    clarity_qualities = df.clarity.unique()
    print("Unique clarity values in the dataframe:", clarity_qualities)
    print()
    
    #Get each unique combo of cut, color and clarity and count the amount of times they occur
    df_by_combo = df.groupby(['cut','color', 'clarity']).size().reset_index().rename(columns={0:'occurances'})
    
    #Select combos which occur more than 800 times
    df_by_combo = df_by_combo[df_by_combo.occurances >= 800]
    
    #Print dataframes with over 800 occurances to console
    print(df_by_combo)
    print()
    
    #List to hold each dataframe containing containing the above combos
    list_df_combo = []
    
    #Extract a new dataframe for each combo from the original dataframe
    for index in df_by_combo.index:
        
        df_new = df[(df['cut'] == df_by_combo['cut'][index]) & (df['color'] == df_by_combo['color'][index]) 
                                                        & (df['clarity'] == df_by_combo['clarity'][index])]
        
        #Extract feature data and target data for each dataframe
        data = np.array(df_new[['carat', 'depth', 'table']])
        target = np.array(df_new['price'])
        
        #Tuple consisting of each combos full dataframe and its features and targets as numpy arrays
        data_tuple = (df_new, data, target)
        
        #Store each new dataframe in a list
        list_df_combo.append(data_tuple)
    
    return list_df_combo
    
#Task 2 - Model Function
def calculate_model_function(deg, data, p):
    
    #Creates numpy array of zeros in the shape of the incoming data
    result = np.zeros(data.shape[0])
    t = 0
    
    #Calculates estimated target vector using tri-variate polynomial to specified degree
    for n in range(deg+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        result += p[t]*(data[:,0]**i)*(data[:,1]**j)*(data[:,2]**k)
                        t = t+1
    return result  

#Function to determine the correct size of the parameter vector from the degree of the multi-variate polynomial
def num_coefficients_3(d):
    
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        t = t+1
    return t     
            
#Task 3 - Linearization
def linearize(deg, data, p0):
    
    #Calculate the polynomial model function
    f0 = calculate_model_function(deg,data,p0)
    
    #Initialize Jacobian matrix to numpy arrays of zeros 
    J = np.zeros((len(f0), len(p0)))
    
    #Epsilon value to compare how the function behaves when epsilon is added to a parameter
    epsilon = 1e-6
    
    for i in range(len(p0)):
        
        #Add parameter vector at index i with the epsilon value
        p0[i] += epsilon
        
        #Calculate the polynomial model function with the changed parameter vectors
        fi = calculate_model_function(deg, data, p0)
        
        #Remove parameter vector at index i with the epsilon value
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:,i] = di
        
    return f0, J

#Task 4 - Parameter Update
def calculate_update(y,f0,J):
    
    l=1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T,r)    
    dp = np.linalg.solve(N,n)    
    
    return dp

#Task 5 - Regression
def regression(deg, train_data, train_target):
    
    #Set max iterations to alternate linearization and parameter update
    max_iter = 10
    
    #Initialize parameter vector coefficients with zeros
    p0 = np.zeros(num_coefficients_3(deg))
    for i in range(max_iter):
        
        f0, J = linearize(deg, train_data, p0)
        dp = calculate_update(train_target, f0, J)
        p0 += dp
        
    return p0
        
#Task 6 - Model Selection
def model_selection(datasets):
    
    #Count value for console display
    count = 1
    
    #List to keep the degree which gave the lowest price difference for each set
    lowest_degree_per_set = []
    
    #Loops over each of the datasets extracted in the pre-processing
    for i in datasets:
        
        #Retrieves and stores the dataframe, feature data and target data from each tuple index
        df = i[0]
        data = i[1]
        target = i[2]
        
        #List to hold which degree gave the lowest price in current dataset
        lowest_degree = []
        
        print("***** DATASET " + str(count) + " RESULTS *****")
        
        #Loop to test degrees 1 to 3 and find which is the optimal degree to use
        for deg in range(1, 4):
            
            #List to hold price differences between each predicted price and actual price in test_target data
            price_diff = []    
    
            #Create k-fold object to split data into 3 different folds
            kfold = KFold(3, shuffle=True, random_state=1)
        
            #K-fold loop for each fold
            for train_index, test_index in kfold.split(df):
                
                #Split the data into train and test data
                train_data = data[train_index]
                train_target = target[train_index]
                test_data = data[test_index]
                test_target = target[test_index]
                
                #Call regression algorithm 
                p0 = regression(deg, train_data, train_target)
                 
                #Get results from polynomial model function, the predicted prices for the test_target data
                res = calculate_model_function(deg, test_data, p0)                
                
                #Loop to compare each predicted price with the actual price in test_target data
                for x in range(len(res)):
                    
                    #Checks which is the higher and the lower price
                    higher_price = max(res[x], test_target[x])
                    lower_price = min(res[x], test_target[x])
                    
                    #Subtracts the lower price from the higher, getting the difference
                    z = higher_price - lower_price
                    
                    #Appends this difference to a list
                    price_diff.append(z)
            
            #Print mean of price difference for degree in set
            print()
            print("MEAN USING DEGREE OF ", deg)
            print(np.mean(price_diff))
            print()
            
            #Append the price difference returned by using each degree to list
            lowest_degree.append(np.mean(price_diff))
          
        #Find lowest price in list and get index plus 1, as this will be the degree which returned smalled price diff for the dataset
        lowest_degree_per_set.append(lowest_degree.index(min(lowest_degree)) + 1)
        
        #Increment count for console display
        count = count + 1
    
    #Print list of degrees which gave the lowest price difference for each set
    print("LIST OF DEGREES RETURNING LOWEST PRICE DIFFERENCE PER SET:", lowest_degree_per_set)
    print()
    
    #Get ooptimal degree and print to console
    optimal_deg = max(set(lowest_degree_per_set), key=lowest_degree_per_set.count)
    print("Optimal Degree: ", optimal_deg)
    
    return lowest_degree_per_set

#Task 7 - Visualization of Results
def visualization_of_results(deg_list, datasets):
    
    #Count for iterating over list of optimal degrees
    count = 0
    
    #Loop over datasets tuple
    for i in datasets:
        
        #Retrieves and stores the dataframe, feature data and target data from each tuple index
        df = i[0]
        data = i[1]
        target = i[2]
        
        #Get optimal degree for current dataset
        deg = deg_list[count]
        
        #Call regression algorithm 
        p0 = regression(deg, data, target)
                           
        #Get results from polynomial model function, the predicted prices for the all target data
        res = calculate_model_function(deg, data, p0)

        #Increment count for optimal list of degrees
        count = count + 1
        
        #Plot actual prices against predicted prices for each dataset
        x = np.array(target)
        plt.xlabel('Actual Prices')
        
        y = np.array(res)
        plt.ylabel('Predicted Prices')
        
        plt.plot(x, y, color='red')
        plt.show()

"""
DRIVER CODE AND MAIN FOR EACH TASK
"""

#Main function
def main():
    
    #Read in excel file to pandas dataframe
    diamonds = pd.read_csv("diamonds.csv")
    
    #Retrieve datasets from pre-processing for later use
    datasets = pre_processing(diamonds)  
    
    #Run the algorithms with the datasets as input
    lowest_degree_per_set = model_selection(datasets)
    
    visualization_of_results(lowest_degree_per_set, datasets)
    
#MAIN DRIVER
main()   