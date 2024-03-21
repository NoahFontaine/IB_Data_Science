import numpy as np
import math
xy_data = np.load('Exercises\Ex1_polyreg_data.npy')

"""
Exercise 1a) Plot the data in a scatterplot
"""

import matplotlib.pyplot as plt
# Your code for scatterplot here

x = xy_data[:,0]  # First column of array (indexed by 0)
y = xy_data[:,1]  # Second column of array (indexed by 1)


# Set parameters to make sure figures are large enough. You can try changing these values
plt.rcParams['figure.figsize'] = [12, 5]
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

plt.scatter(x, y, s=10)   # s can be used to adjust the size of the dots
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Plot of the scatterplot')
plt.show()


"""
Exercise 1b) Write a function `polyreg` to fit a polynomial of a given order to a dataset. 
"""

def polyreg(data_matrix, k):
    # Your code here
    x = data_matrix[:,0]
    y = data_matrix[:,1]
    
    X_t = []                # Create empty X matrix and append all the values to the kth order
    if k >= np.size(x):
        for n in range(np.size(x)-2):
            X_t.append(np.zeros(np.size(x))**n)
        X_t.append(x**(np.size(x)-1))
    else:
        for n in range(k+1):
            X_t.append(x**n)
    
    XT = np.array(X_t)
    X = np.transpose(X_t)   # X matrix with N rows and k+1 columns

    # Calculate beta vector and print its coefficients
    beta_vec = np.linalg.inv(XT.dot(X)).dot(XT.dot(y))
    print("Regression coefficients for polynomial model:")
    for i in range(k):
        print("\t beta_"+str(i)+" = "+str(np.round(beta_vec[i], decimals=3)))

    # Compute the SSE, which is SSE = ||y - X*beta||**2:
    SSE = np.linalg.norm(y - X.dot(beta_vec))**2
    print("SSE of the polynomial model:", np.round(SSE, decimals=3))

    # Compute the vector of residuals:
    resid_vec = y - X.dot(beta_vec)
    print("Vector of residuals of the polynomial model:")
    print(np.round(resid_vec, decimals=3))


    # The function should return the the coefficient vector beta, the fit, and the vector of residuals
    return(beta_vec, SSE, resid_vec)

polyreg(xy_data, 7)


"""
Exercise 1c) Use `polyreg` to fit polynomial models for the data in `xy_data` for $k=2,3,4$:
"""

# Points of the data
x = xy_data[:,0]  # First column of array (indexed by 0)
y = xy_data[:,1]  # Second column of array (indexed by 1)

# Plot of the poly model for k = 2, with the same x_coordinates as data
beta_vec_k2 = polyreg(xy_data, 2)[0]
y_k2 = beta_vec_k2[0] + beta_vec_k2[1]*x + beta_vec_k2[2]*x**2

# Poly model for k = 3
beta_vec_k3 = polyreg(xy_data, 3)[0]
y_k3 = beta_vec_k3[0] + beta_vec_k3[1]*x + beta_vec_k3[2]*x**2 + beta_vec_k3[3]*x**3

# Poly model for k = 4
beta_vec_k4 = polyreg(xy_data, 4)[0]
y_k4 = beta_vec_k4[0] + beta_vec_k4[1]*x + beta_vec_k4[2]*x**2 + beta_vec_k4[3]*x**3 + beta_vec_k4[4]*x**4

# SSE for the models and SSE_0
SSE_0 = polyreg(xy_data, 0)[1]
SSE_2 = polyreg(xy_data, 2)[1]
SSE_3 = polyreg(xy_data, 3)[1]
SSE_4 = polyreg(xy_data, 4)[1]

print("The SSEs for the polynomial models with k = (2,3,4) are (" + str(np.round(SSE_2, 3)) + ", " + str(np.round(SSE_3, 3)) + ", " + str(np.round(SSE_4, 3)) + ") respectively.")

# Calculate R^2 values
R_2 = np.round(1-SSE_2/SSE_0, 3)
R_3 = np.round(1-SSE_3/SSE_0, 3)
R_4 = np.round(1-SSE_4/SSE_0, 3)

print("The R^2 values for the polynomial models with k = (2,3,4) are (" + str(R_2) + ", " + str(R_3) + ", " + str(R_4) + ") respectively.")

plt.scatter(x, y, s=10)
plt.plot(x, y_k2, "-", label = "$k = 2$")
plt.plot(x, y_k3, "-", label = "$k = 3$")
plt.plot(x, y_k4, "-", label = "$k = 4$")
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Plot of the scatterplot with the polynomial models for $k = n$')
plt.legend()
plt.savefig('Ex1c_plot.pdf', bbox_inches = 'tight')
plt.show()


"""
Exercise 1d) For the model you have chosen in the previous part (either $k=2/3/4)$:
"""

# Scatter plot of the residuals
plt.rcParams['figure.figsize'] = [10, 5]
plt.scatter(x, polyreg(xy_data, 3)[2])
plt.title("Scatter plot of the residuals for $k=3$")
plt.xlabel("x-values")
plt.ylabel("Residuals")
plt.show()

# Histogram of the residuals with the Gaussian pdf
print('Mean of residuals = ', np.round(np.mean(polyreg(xy_data, 3)[2]), 5), 'Variance of residuals = ', 
      np.round(np.var(polyreg(xy_data, 3)[2]), 3))

# Plot normed histogram of the residuals
n, bins, patches = plt.hist(polyreg(xy_data, 3)[2], bins=14, density=True, facecolor='green');

# Plot Gaussian pdf with same mean and variance as the residuals
from scipy.stats import norm

res_stdev = np.std(polyreg(xy_data, 3)[2])  #standard deviation of residuals
xvals = np.linspace(-3*res_stdev,3*res_stdev,1000)
plt.plot(xvals, norm.pdf(xvals, loc=0, scale=res_stdev), 'r')
plt.xlabel("Residuals")
plt.ylabel("Relative frequency")
plt.title("Histogram of the residuals for $k=3$ with its Gaussian pdf")
plt.show()