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
