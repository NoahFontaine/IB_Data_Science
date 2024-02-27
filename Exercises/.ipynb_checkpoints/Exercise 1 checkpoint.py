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
plt.savefig('Ex1a_plot.pdf', bbox_inches = 'tight')
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