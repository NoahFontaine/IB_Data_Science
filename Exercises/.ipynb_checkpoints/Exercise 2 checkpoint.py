import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_hosp = pd.read_csv('hospital_cases_2023-02-16.csv')  # Create a data frame by loading data from a csv file
# If running on Google Colab change path to '/content/drive/MyDrive/IB-Data-Science/Exercises/hospital_cases_2023-02-16.csv'

df_hosp.head(3)   #display the first three rows

df_hosp.tail(3)

plt.rcParams['figure.figsize'] = [14, 7]
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

df_hosp.plot(x='date', y='hospitalCases')
plt.show()

df_part = df_hosp[635:656]
df_part.head(2)

df_part.tail(2)

yvals = np.array(df_part['hospitalCases'])
N = np.size(yvals)
xvals = np.linspace(1,N,N) #an array containing the values 1,2....,N


"""
Exercise 2a) Plot the data in a scatterplot:
"""

# Your code for scatterplot here
plt.scatter(xvals, yvals, s=10)
plt.xlabel('Days since 21/12/2022')
plt.ylabel('Number of hospital cases')
plt.xticks(np.arange(0, max(xvals)+1, 2))
plt.title('Scatterplot of the number of COVID hospital cases in late 2021/early 2022')


"""
Exercise 2b) Fit an exponential model to the data
"""

# Form the X matrix with columns 1 and x for x in dates.
all_ones = np.ones(np.shape(xvals))
X = np.column_stack((all_ones, xvals))
XT = np.transpose(X)

# Calculate the beta vector, where beta_vec = [log(c1), c2] and print it
beta_vec = np.linalg.inv(XT.dot(X)).dot(XT.dot(np.log(yvals)))
print("The value of c1 is", np.round(np.exp(beta_vec[0]),3), "and the value of c2 is", np.round(beta_vec[1], 3))

#Plot the scatter plot with the fit
y = np.exp(beta_vec[0])*np.exp(beta_vec[1]*xvals) # y-values of the model

plt.scatter(xvals, yvals, s=10)
plt.plot(xvals, y, "-", label = "$y = c_1 e^{c_2x}$")
plt.xlabel('Days since 21/12/2022')
plt.ylabel('Number of hospital cases')
plt.xticks(np.arange(0, max(xvals)+1, 2))
plt.legend()
plt.title('Scatterplot of the COVID hospital cases with the exponential model')
plt.show()


"""
Exercise 2c) Estimate the weekly growth rate in hospital admissions over this period
"""

# compute and print weekly growth rate (in %)
print("The percentage weekly growth rate in hospital admissions over this period is approximately", np.round(100*np.exp(7*beta_vec[1]), 3), "%")