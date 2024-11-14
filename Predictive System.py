# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:29:49 2024

@author: suhas
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('D:\Mini-Project-CKDP/trained_model.sav', 'rb'))


input_data = (6,85,66,29,0,26.6,0.351,31)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person does not have CKD')
else:
  print('The person has been diagnosed with CKD')