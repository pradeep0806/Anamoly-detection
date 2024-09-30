'''
Z-Test
'''

from scipy import  stats
import numpy as np

np.random.seed(0)
grp1 = np.random.normal(0.1,1.0,1000)
grp2 = np.random.normal(0.2,1.0,1000)

z_score,p_value = stats.ttest_ind(grp1,grp2)
print(f"z-score :{z_score},p-value :{p_value}")

'''
The z-score is a measure of how many standard deviations a data point is from the mean of the dataset. It’s calculated as:

z = X-𝜇/σ 
 
Where:
X is the data point.
μ is the mean of the dataset.
σ is the standard deviation of the dataset.

The p-value is the probability that the observed data point would be as extreme as or more extreme than the observed value, under the assumption that the data follows a normal distribution.
A small p-value (e.g., less than 0.05) suggests that the data point is statistically significant and is likely an anomaly.
In the context of anomalies, a low p-value indicates that it’s unlikely the data point belongs to the same distribution as the rest of the data.

When to Use a Z-Test:

Two-Sample Z-Test is used when:
You have two independent samples.
The population variances are known and assumed to be equal.
The sample size is large enough (typically 
𝑛>30
n>30 for each group) to assume a normal distribution of the sample means (Central Limit Theorem).
If the population variances are not known or the sample size is small, a t-test (specifically, a two-sample t-test) is more appropriate.
'''

""" T-Test"""

grp1 = np.random.normal(0.1,1.0,30)
grp2 = np.random.normal(0.2,1.0,30)
t_score , p_value = stats.ttest_ind(grp1,grp2)
print(f"t-score :{t_score},p-value :{p_value}")

"""same as z-test but for small sample size"""


"""without a specific distribution of data we can use non parametric tests as chi-squared tests, less powerful but more flexible"""

'''
Grubbs test to determine a outlier
'''

data = np.random.normal(0,1,100)
data[15] = 3.5
data[30] = 5

def grubbs_test(data,alpha = 0.05):
    n=len(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = (data-mean)/std_dev
    abs_z_scores = np.abs((data-mean)/std_dev)
    max_z_score = np.max(abs_z_scores)
    max_z_score_index = np.argmax(abs_z_scores)

    print(f"number of data points:{n}")
    print(f"mean of data : {mean}")
    print(f"standard deviation of data : {std_dev}")
    print("\nZ-scores ")
    print(z_scores)
    print(f"Max Absolute Z-Score: {max_z_score} located at index {max_z_score_index}\n")
    g_calculated = (n-1)*np.sqrt(max_z_score**2 / (n-2+max_z_score**2))
    t_critical = stats.t.ppf(1-alpha/(2*n),n-2)
    g_critical = ((n-1)/np.sqrt(n))*np.sqrt(t_critical**2/(n-2 + t_critical**2))

    # If the calculated test statistic is larger than the critical value, then reject the null hypothesis and mark the data point as an outlier
    if g_calculated > g_critical:
        print(f"Outlier detected at index {max_z_score_index}.")
        outlier = True
    else:
        print("No outliers detected.")
        outlier = False
    return max_z_score_index, outlier

outlier_index, is_outlier = grubbs_test(data)