""" Anamoly in 3 or more features"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance

def calcmahalanobis(data):
    robust_covariance = EmpiricalCovariance().fit(data)
    m_dist = robust_covariance.mahalanobis(data)
    return m_dist

# Define multivariate data
data = np.array([[100000, 16000, 300, 60, 76],
                [800000, 60000, 400, 88, 89],
                [650000, 300000, 1230, 90, 89],
                [700000, 10000, 300, 87, 57],
                [860000, 252000, 400, 83, 79],
                [730000, 350000, 104, 81, 84],
                [400000, 260000, 632, 72, 78],
                [20, 260000, 6302, 2, 2000],
                [870000, 510000, 221, 91, 99],
                [780000, 2000, 142, 90, 97],
                [400000, 5000, 267, 93, 99]])
columns = ['Price','Distance','Emission','Performance','Mileage']
df = pd.DataFrame(data,columns=columns)

print(f"Data : \n{df}\n")
m_dist = calcmahalanobis(df)

print(f"Mahalanobis Distance :\n{m_dist}\n")

alpha =0.10

thresh = chi2.ppf(1-alpha,df = df.shape[1])
print(f"Threshold for outlier detection: {thresh}\n")
outliers = df[m_dist>thresh]
df['distance'] = m_dist
df['outlier'] = df['distance'] > thresh
print(f"Outliers:\n{outliers}")
print(f"Data:\n{df}\n")

'''Mahalanobis Distance is a measure of the distance between a point and a distribution. It differs from Euclidean distance as it considers the correlations between variables in the dataset.
This distance measures how far away a data point is from the center (mean) of the distribution, considering the covariance structure of the data. Essentially, it tells you how unusual or different a point is from the "typical" data in the set.'''