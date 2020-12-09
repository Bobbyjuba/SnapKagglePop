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

import math
import pandas as pd
import scipy.stats as stats
import numpy as np

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