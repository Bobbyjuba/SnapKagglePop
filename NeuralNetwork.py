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

# Change Male into 0 and Female into 1
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

''' Gender '''
# genders = df["Gender"].apply(GenderOneHot)
# genders.to_csv("genders.csv", index = False)

''' Age '''
# ages = df["Age"]
# new_ages = stats.zscore(ages)
# df_ages = pd.DataFrame(new_ages, columns=['Age'])

''' Driving License '''
# driving_license = df["Driving_License"]
# driving_license.to_csv('licenses.csv', index = False)

''' Region Code '''
# regions = df["Region_Code"]
# new_regions = stats.zscore(regions)
# df_regions = pd.DataFrame(new_regions, columns=['Region Code'])

''' Vehicle Age '''
# vehicle_age = df["Vehicle_Age"].apply(VehicleAgeOneHot)
# vehicle_age.to_csv('vehicle_ages.csv', index = False)

''' Vehicle Damage '''
# vehicle_damage = df["Vehicle_Damage"].apply(VehicleDamageOneHot)
# vehicle_damage.to_csv('vehicle_damages.csv', index = False)

''' Annual Premium '''
# premiums = df["Annual_Premium"]
# new_premiums = stats.zscore(premiums)
# df_premiums = pd.DataFrame(new_premiums, columns=['Annual Premium'])

''' Policy Sales Channel '''
# channel = df["Policy_Sales_Channel"]
# new_channel = stats.zscore(channel)
# df_channel = pd.DataFrame(new_channel, columns=['Policy Sales Channel'])

''' Vintage '''
# vintage = df["Vintage"]
# new_vintage = stats.zscore(vintage)
# df_vintage = pd.DataFrame(new_vintage, columns=['Vintage'])