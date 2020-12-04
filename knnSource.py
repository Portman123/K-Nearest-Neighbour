# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import math
import numpy as np
import matplotlib.pyplot as plt 


def splitData():
    #Returns split data, ready for f2 and f3
    
    digits = datasets.load_digits()
    
    #Splits data into training and test with 80% training, 20% test
    return(train_test_split(digits.data,digits.target, test_size=0.2))


#FUNCTIONALITY 1
def f1():
    
    print('==============================================================================================')
    print('DATA INFORMATION')
    
    digits = datasets.load_digits()
    
    #Number of Data Entries
    print('')
    print('number of digit entries: ', digits.images.shape)
    print('i.e. 1797 8x8 matrices')
    
    #Number of Classes
    print('')
    print("There are 10 Classes: ", digits.target_names)
    
    data_entries = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for name in digits.target_names:
        for item in digits.target:
            if name == item:
                data_entries[name] += 1
    print('')
    print('data entries for each class: ', data_entries)
    
    #Min and Max Values for Each Feature
    
    #Train and Test Dataset Split
    print('')
    print('I\'ve chosen to split my data into 80% training, 20% test')
    print('')
    
    
#FUNCTIONALITY 2
def f2(data_train, data_test, target_train, target_test):
    #train model and make predictions using anything form the anaconda package
    
    #Train the model 
    clf = KNeighborsClassifier(round(math.sqrt(len(data_train)))).fit(data_train,target_train)
    
    #Make predictions on test data
    predictions = clf.predict(data_test)
    
    return(metrics.accuracy_score(target_test, predictions))
    
    
#FUNCTIONALITY 3
def f3(data_train, data_test, target_train, target_test):
    #create implementation without using machine learning algorithms in the anaconda package
    
    #put all training data into one array
    #(first element is data, second is target)
    train = []
    count = 0
    for item in data_train:
        train.append([item, target_train[count]])
        count+=1
        
    #put all test data into one array
    #(first element is data, second is target)
    test = []
    count = 0
    for item in data_test:
        test.append([item, target_test[count]])
        count+=1
    
    #work out how many predictions were correct from tests
    correct = 0
    for item in test:
        if knn(round(math.sqrt(len(data_train))), train, item) == item[1]:
            correct += 1
    
    #return accuracy
    return(correct/len(test))
            

#PART OF F3
def knn(k, data, prediction):
    #Takes k, training data and data to be predicted
    #Returns predicted digit for 1 test instance
    
    distances = []
    
    #Calculate euclidean distances between every trained digit_data and 1 test digit_data
    #Store in array with corresponding digits_target value
    for item in data:
        distances.append([np.linalg.norm(item[0] - prediction[0]), item[1]])

    nearest = []
    
    #Find smallest Distances with their corresponding digits_target value and store in array of length k 
    for i in range(k):
        closest = [distances[0],0]
        count = 0
        for item in distances:
            if item[0] <= closest[0][0]:
                closest = [item, count]
            count += 1
        distances.remove(distances[closest[1]])
        nearest.append(closest[0])
        
    #Determin frequency of each new nearest neighbor
    votes = {}
    for item in nearest:
        if item[1] not in votes:
            votes[item[1]] = 1
        else:
            votes[item[1]] += 1
            
    #Determin most frequent nearest neighbor
    prediction = [0,0]
    for key in votes:
        if votes[key] >= prediction[1]:
            prediction = [key, votes[key]]
    
    #Return as prediction
    return(prediction[0])
    

#FUNCTIONALITY 4
def f4():
    #Calls functions f2,f3 to compare accuracy over same data splits
    #Displays calculated data
    
    #call f2 and f3 using the data above
    comparrison_accuracy_f2 = f2(comparrison_data_train, comparrison_data_test, comparrison_target_train, comparrison_target_test)
    comparrison_accuracy_f3 = f3(comparrison_data_train, comparrison_data_test, comparrison_target_train, comparrison_target_test)
    
    #Display
    print('==============================================================================================')
    print('')
    print('I used the knn learning algorithm so there are no saved models...')
    print('...The following are the two algortihms using the same data values...')
    print('---------------------------------------------------------------------------------------------')
    print('')
    print('Percentage accuracy of Predictions using SciKitLearn\'s built in knn learning algorithm... ')
    print('...followed by percentage accuracy of my own implementation of the knn learning algorithm...')
    print('...all values rounded to 2d.p....')
    print('')
    print('f2:.......... ', round((comparrison_accuracy_f2 * 100), 2),'%')
    print('f3:.......... ', round((comparrison_accuracy_f3 * 100), 2),'%')
    print('')
    print('Precentage Error rounded to 2d.p....')
    print('')
    print('f2:.......... ', round(((1-comparrison_accuracy_f2)*100), 2),'%')
    print('f3:.......... ', round(((1-comparrison_accuracy_f3)*100), 2),'%')
    
    
def f5():
    
    #Display instructions
    print('==============================================================================================')
    print('INPUT "x" to return to main menu')
    print('Input the index of the test prediction you\'d like to see')
    print('(', len(comparrison_target_test),'predictions were calculated)')
    print('(Remember that  0  is the first element and ', (len(comparrison_target_test)-1), ' is the last)')
    print('')
    userInput = input('')
    print('')
    
    #Interpret User's input
    if userInput == 'x':
        mainMenu()
    else:
        userInput = int(userInput)
        
        #Show image of number being predicted with target number
        print('Number to be predicted: ', comparrison_target_test[userInput])
        print('')
        plt.gray()
        plt.matshow(np.reshape(comparrison_data_test[userInput], (8,8)))
        plt.show()
        
        #Show f2s prediction
        print('')
        clf = KNeighborsClassifier(round(math.sqrt(len(comparrison_data_train)))).fit(comparrison_data_train,comparrison_target_train)
        print('f2\'s prediction: ', clf.predict(([comparrison_data_test[userInput]])))
        
        #put all training data into one array
        #(first element is data, second is target)
        train = []
        count = 0
        for item in comparrison_data_train:
            train.append([item, comparrison_target_train[count]])
            count+=1
        
        #Show f3s prediction
        print('')
        print('f3\'s prediction: ', knn(round(math.sqrt(len(comparrison_data_train))), train, [comparrison_data_test[userInput], comparrison_target_test[userInput]]))
            
        #repeate f5 until user quits
        f5()

def createNewDataSplit():
    global comparrison_data_train, comparrison_data_test, comparrison_target_train, comparrison_target_test
    
    #Splits data into training and test
    comparrison_data_train, comparrison_data_test, comparrison_target_train, comparrison_target_test = splitData()

def mainMenu():
    
    #Display Instructions
    print('==============================================================================================')
    print('')
    print('Main Menu')
    print('')
    print('1. f1: Data set info ')
    print('2. f4: Compare prediciton accuracies of f2,f3')
    print('3. f5: Query predicitions')
    print('4. Create new data split to test')
    print('X. Exit')
    print('')
    print('Enter options: "1", "2", "3", "4" or "x".')
    userInput = input('')
    print('')
    
    #Interpret User's input
    if userInput == '1':
        f1()
        mainMenu()
    elif userInput == '2':
        f4()
        mainMenu()
    elif userInput == '3':
        f5()
    elif userInput == '4':
        createNewDataSplit()
        print('')
        print('NEW DATA SPLIT CREATED')
        print('')
        mainMenu()
    elif userInput == 'x':
        print('Exit')
    else:
        print('Invalid input. Enter options: "1", "2", "3" or "x". ')
        mainMenu()
    
#Set Initial data split    
comparrison_data_train, comparrison_data_test, comparrison_target_train, comparrison_target_test = splitData()

mainMenu()