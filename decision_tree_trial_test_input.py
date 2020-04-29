# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:03:10 2020

@author: Dr_programmer
"""

import pandas as pd
import joblib
from decimal import getcontext, Decimal




modelFile = joblib.load('clf_GiniModel.model')

#test = [[4,86.49572832,82.33190465,7.78,0,9,6,9,8,8,9]]
#test = [[1,63.9605725,64.05273927,7.78,3,7,6,8,6,9,8]]
#test = [[4,74.55286199,77.14460231,7.36,0,6,5,8,7,8,8]]
#test = [[9,12,7,12,6.53,3.2,6.7,5.9,4.6,4,5.3]] #Row 16 --- 0
test = [[7,14,7,11,8.99,8,8,8,8,8,9.6]]  #Row 15 --- 1
#test = [[12,7,9,14,6.54,4.3,6,5.8,2.5,8,4.6]] #Row 55 --- 0
testResult = modelFile.predict(test)

def numbers_to_strings(argument): 
    switcher = { 
        1: "The system predicts that the student is going to get placed with a probability of : ",
        0: "The system predicts that the student may not get placed with a probablity of : ", 
    }
    return switcher.get(argument, "nothing")

def numbers_to_strings_alternate(argument): 
    switcher = { 
        0: "The system predicts that the student is going to get placed with a probability of : ",
        1: "The system predicts that the student may not get placed with a probablity of : ", 
    }
    return switcher.get(argument, "nothing")
  
result = numbers_to_strings(testResult[0])

# Set the precision.
probability = str(modelFile.predict_proba(test)[0][0])
probability = probability[0:3]
probability = float(probability)*100

print(probability)

if probability < 45 and testResult[0] == 0:
  probability = 100-probability
  result = numbers_to_strings_alternate(testResult[0])
  

print(result,probability,"%")

'''with open('out.txt','w') as output:
  output.write(result)'''
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  