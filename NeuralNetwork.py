######################################################################
#                                                                    #
#  CS 461 Assignment 3                                Brian Roden    #
#                                                                    #
#                                                                    #
#                                                                    #
#  12/8/2020                                          12/11/2020     #
#                                                                    #
#                                                                    #
######################################################################

import math
import pandas as pd
import scipy.stats as stats
import numpy as np
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def RemoveID(ID):
    return ''

def GenderOneHot(gender):
    if gender == 'Male':
        return 0

    else:
        return 1

def VehicleAgeOneHot(vehicleAge):
    if vehicleAge == '> 2 Years':
        return 1

    elif vehicleAge == '1-2 Year':
        return 0.5

    else:
        return 0

def VehicleDamageOneHot(vehicleDamage):
    if vehicleDamage == 'Yes':
        return 1

    else:
        return 0

df = pd.read_csv('training_data.csv')

''' ID '''
df = df.drop(['id'], axis = 1)

''' Gender '''
df["Gender"] = df["Gender"].apply(GenderOneHot)

''' Age '''
ages = df["Age"]
new_ages = stats.zscore(ages)
df["Age"] = new_ages

''' Region Code '''
regions = df["Region_Code"]
new_regions = stats.zscore(regions)
df["Region_Code"] = new_regions

''' Vehicle Age '''
df["Vehicle_Age"] = df["Vehicle_Age"].apply(VehicleAgeOneHot)

''' Vehicle Damage '''
df["Vehicle_Damage"] = df["Vehicle_Damage"].apply(VehicleDamageOneHot)


''' Annual Premium '''
premiums = df["Annual_Premium"]
new_premiums = stats.zscore(premiums)
df["Annual_Premium"] = new_premiums

''' Policy Sales Channel '''
channel = df["Policy_Sales_Channel"]
new_channel = stats.zscore(channel)
df["Policy_Sales_Channel"] = new_channel

''' Vintage '''
vintage = df["Vintage"]
new_vintage = stats.zscore(vintage)
df["Vintage"] = new_vintage

# df.to_csv('new_data.csv', index = False)

x = df.drop(['Response'], axis=1)
y = df['Response']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15, random_state = 52)

nn = Sequential()
nn.add(Dense(1, input_dim = 10, activation='sigmoid'))
# nn.add(Dense(1, activation='sigmoid'))

nn.compile(loss='binary_crossentropy', optimizer='adam')
nn.fit(X_train, Y_train, epochs=3)

print(nn.summary())
scores = nn.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores * 100))