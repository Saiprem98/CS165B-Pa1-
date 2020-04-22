from __future__ import division
import numpy as np
def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    # TODO: IMPLEMENT

    train_count = training_input[0]
    train_data = np.array(training_input[1:])
    
    #remove first line to take input into for loops 
    test = testing_input.pop(0)
    test.pop(0)
    ABCtotal = test[0]


    A_train_data = train_data[:train_count[1]]
    B_train_data = train_data[train_count[1]:train_count[1]+train_count[2]]
    C_train_data = train_data[train_count[1]+train_count[2]:]



    classCount = training_input[0][0]
    # values in center array [None, None ...... num_A] 
    center_A = [None]*classCount
    center_B = [None]*classCount
    center_C = [None]*classCount 
 
    center_A = np.mean(A_train_data, axis=0)
    center_B = np.mean(B_train_data, axis=0)
    center_C = np.mean(C_train_data, axis=0)

   
        
    # calculate w vector for dot product 
    # w = p - n
    # t = 1/2(p-n)^T(p-n)
    # t = 1/2(w)^2(p-n)
    # x*w > t
    AtoB = [None] * classCount
    AtoC = [None] * classCount
    BtoC = [None] * classCount
    distance_A = 0
    distance_B = 0
    distance_C = 0

    for i in range(0, classCount):
        AtoB[i] = center_A[i] - center_B[i]
        AtoC[i] = center_A[i] - center_C[i]
        BtoC[i] = center_B[i] - center_C[i]
        distance_A = distance_A + (center_A[i]**2)
        distance_B = distance_B + (center_B[i]**2)
        distance_C = distance_C + (center_C[i]**2)

    # t = 1/2(p-n)^T(p-n)
    t_AtoB = (distance_A - distance_B)/2
    t_AtoC = (distance_A - distance_C)/2
    t_BtoC = (distance_B - distance_C)/2




    true_postive_A = 0
    false_postive_A = 0
    true_negative_A = 0
    false_negative_A = 0

    true_postive_B = 0
    false_postive_B = 0
    true_negative_B = 0
    false_negative_B = 0

    true_postive_C = 0
    false_postive_C = 0
    true_negative_C = 0
    false_negative_C = 0

    # input x and dot product with w
    # check if greater or less than t 
    # see which class it gets classified to and check if correctly classified 
    # track values on chart
    testingTotal =  ABCtotal*3 #75
    for i in range(0,3):
        for j in range(0, testingTotal): # 0->75
            x = testing_input[j]
            if(np.dot(AtoB,x) >= t_AtoB): # Check A vs. B 
                if(np.dot(AtoC, x) >= t_AtoC): #Check A vs. C
                     #Machine predicted A 
                    if(j >= 0 and j < ABCtotal): # First A data
                        if(i == 0): # true positive A 
                            true_postive_A += 1
                        if(i == 1): # not A true negative B did not predict 
                            true_negative_B += 1
                        if(i == 2): # not A true negative C
                            true_negative_C += 1
                    if(j >= ABCtotal and j < ABCtotal * 2): # B data 
                        if(i == 0): # false + A
                            false_postive_A +=1 
                        if(i == 1): # false - B
                            false_negative_B +=1
                        if(i == 2): # true - C 
                            true_negative_C += 1
                    elif(j >= ABCtotal * 2 and j < ABCtotal * 3): # C data 
                        if(i == 0):
                            false_postive_A +=1
                        if(i == 1):
                            true_negative_B +=1
                        if(i == 2):
                            false_negative_C +=1
            else: 
                if(np.dot(BtoC, x) >= t_BtoC):
                    # Machine preditced B 
                    if(j >= 0 and j < ABCtotal):
                        if(i == 0): 
                            false_negative_A +=1
                        if(i == 1): 
                            false_postive_B +=1
                        if(i == 2):
                            true_negative_C +=1
                    if(j >= ABCtotal and j < ABCtotal * 2):
                        if(i == 0):
                            true_negative_A +=1
                        if(i == 1):
                            true_postive_B +=1
                        if(i == 2):
                            true_negative_C +=1
                    elif(j >= ABCtotal * 2 and j < ABCtotal * 3):
                        if(i == 0):
                            true_negative_A +=1
                        if(i == 1):
                            false_postive_B +=1
                        if(i == 2):
                            false_negative_C +=1

                else: 
                    # last choice has to be C 
                    if(j >= 0 and j < ABCtotal):
                        if(i == 0): 
                            false_negative_A +=1
                        if(i == 1): 
                            true_negative_B +=1
                        if(i == 2): 
                            false_postive_C +=1
                    if(j >= ABCtotal and j < ABCtotal * 2):
                        if(i == 0):
                            true_negative_A +=1
                        elif(i == 1):
                            false_negative_B +=1
                        elif(i == 2):
                            false_postive_C +=1
                    elif(j >= ABCtotal * 2 and j < ABCtotal * 3):
                        if(i == 0):
                            true_negative_A +=1
                        if(i == 1):
                            true_negative_B +=1
                        if(i == 2):
                            true_postive_C +=1

    true_postive_total = (true_postive_A + true_postive_B + true_postive_C)/3
    true_negative_total= (true_negative_A + true_negative_B + true_negative_C)/3
    false_postive_total = (false_postive_A + false_postive_B + false_postive_C)/3
    false_negative_total = (false_negative_A + false_negative_B + false_negative_C)/3

    
    total_P = test[0]
    total_N = 2*test[0] 
    estimate_P = true_postive_total + false_postive_total

    true_positive_rate = true_postive_total/total_P
    false_positive_rate = false_postive_total/total_N
    error_rate = (false_postive_total+ false_negative_total)/(total_P+total_N)
    accuracy = 1 - error_rate
    precision = true_postive_total/estimate_P

    results = {
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "precision": precision,
    }

    return results

   
