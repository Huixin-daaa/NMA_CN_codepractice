# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# @title Plotting Functions

def plot_data(X):
  """
  Plots bivariate data. Includes a plot of each random variable, and a scatter
  plot of their joint activity. The title indicates the sample correlation
  calculated from the data.

  Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable

  Returns:
    Nothing.
  """

  fig = plt.figure(figsize=[8, 4])
  gs = fig.add_gridspec(2, 2)
  ax1 = fig.add_subplot(gs[0, 0])
  ax1.plot(X[:, 0], color='k')
  plt.ylabel('Neuron 1')
  plt.title(f'Sample var 1: {np.var(X[:, 0]):.1f}')
  ax1.set_xticklabels([])
  ax2 = fig.add_subplot(gs[1, 0])
  ax2.plot(X[:, 1], color='k')
  plt.xlabel('Sample Number')
  plt.ylabel('Neuron 2')
  plt.title(f'Sample var 2: {np.var(X[:, 1]):.1f}')
  ax3 = fig.add_subplot(gs[:, 1])
  ax3.plot(X[:, 0], X[:, 1], '.', markerfacecolor=[.5, .5, .5],
           markeredgewidth=0)
  ax3.axis('equal')
  plt.xlabel('Neuron 1 activity')
  plt.ylabel('Neuron 2 activity')
  plt.title(f'Sample corr: {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:.1f}')
  plt.show()


def plot_basis_vectors(X, W):
  """
  Plots bivariate data as well as new basis vectors.

  Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable
    W (numpy array of floats) : Square matrix representing new orthonormal
                                basis each column represents a basis vector

  Returns:
    Nothing.
  """

  plt.figure(figsize=[4, 4])
  plt.plot(X[:, 0], X[:, 1], '.', color=[.5, .5, .5], label='Data')
  plt.axis('equal')
  plt.xlabel('Neuron 1 activity')
  plt.ylabel('Neuron 2 activity')
  plt.plot([0, W[0, 0]], [0, W[1, 0]], color='r', linewidth=3,
           label='Basis vector 1')
  plt.plot([0, W[0, 1]], [0, W[1, 1]], color='b', linewidth=3,
           label='Basis vector 2')
  plt.legend()
  plt.show()


def plot_data_new_basis(Y):
  """
  Plots bivariate data after transformation to new bases.
  Similar to plot_data but with colors corresponding to projections onto
  basis 1 (red) and basis 2 (blue). The title indicates the sample correlation
  calculated from the data.

  Note that samples are re-sorted in ascending order for the first
  random variable.

  Args:
    Y (numpy array of floats) : Data matrix in new basis each column
                                corresponds to a different random variable

  Returns:
    Nothing.
  """
  fig = plt.figure(figsize=[8, 4])
  gs = fig.add_gridspec(2, 2)
  ax1 = fig.add_subplot(gs[0, 0])
  ax1.plot(Y[:, 0], 'r')
  plt.xlabel
  plt.ylabel('Projection \n basis vector 1')
  plt.title(f'Sample var 1: {np.var(Y[:, 0]):.1f}')
  ax1.set_xticklabels([])
  ax2 = fig.add_subplot(gs[1, 0])
  ax2.plot(Y[:, 1], 'b')
  plt.xlabel('Sample number')
  plt.ylabel('Projection \n basis vector 2')
  plt.title(f'Sample var 2: {np.var(Y[:, 1]):.1f}')
  ax3 = fig.add_subplot(gs[:, 1])
  ax3.plot(Y[:, 0], Y[:, 1], '.', color=[.5, .5, .5])
  ax3.axis('equal')
  plt.xlabel('Projection basis vector 1')
  plt.ylabel('Projection basis vector 2')
  plt.title(f'Sample corr: {np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]:.1f}')
  plt.show()

# %% [markdown]
# ## Generate correlated multivariate data

# %%
# @markdown Execute this cell to get helper function `get_data`

def get_data(cov_matrix):
  """
  Returns a matrix of 1000 samples from a bivariate, zero-mean Gaussian.

  Note that samples are sorted in ascending order for the first random variable

  Args:
    cov_matrix (numpy array of floats): desired covariance matrix

  Returns:
    (numpy array of floats) : samples from the bivariate Gaussian, with each
                              column corresponding to a different random
                              variable
  """

  mean = np.array([0, 0])
  X = np.random.multivariate_normal(mean, cov_matrix, size=1000)
  indices_for_sorting = np.argsort(X[:, 0])
  X = X[indices_for_sorting, :]

  return X

help(get_data)

# %%
def calculate_cov_matrix(var_1, var_2, corr_coef):
  """
  Calculates the covariance matrix based on the variances and correlation
  coefficient.

  Args:
    var_1 (scalar)          : variance of the first random variable
    var_2 (scalar)          : variance of the second random variable
    corr_coef (scalar)      : correlation coefficient

  Returns:
    (numpy array of floats) : covariance matrix
  """

  #################################################
  ## TODO for students: calculate the covariance matrix
  # Fill out function and remove
  #raise NotImplementedError("Student exercise: calculate the covariance matrix!")
  #################################################

  # Calculate the covariance from the variances and correlation
  cov = corr_coef*(var_1**(1/2))*(var_2**(1/2))

  cov_matrix = np.array([[var_1, cov], [cov, var_2]])

  return cov_matrix


# Set parameters
np.random.seed(2020)  # set random seed
variance_1 = 1
variance_2 = 1
corr_coef = 0.8

# Compute covariance matrix
cov_matrix = calculate_cov_matrix(variance_1, variance_2, corr_coef)

# Generate data with this covariance matrix
X = get_data(cov_matrix)

# Visualize
plot_data(X)

# %% [markdown]
# ## Define a new orthonormal basis

# %% [markdown]
# In two dimensions, it is easy to make an arbitrary orthonormal basis. All we need is a random vector u, which we have normalized. We can now define the second basis vector to be w=[−u2,u1], and [u w] will be an orthnormal basis.

# %%
def define_orthonormal_basis(u):
  """
  Calculates an orthonormal basis given an arbitrary vector u.

  Args:
    u (numpy array of floats) : arbitrary 2-dimensional vector used for new
                                basis

  Returns:
    (numpy array of floats)   : new orthonormal basis
                                columns correspond to basis vectors
  """

  #################################################
  ## TODO for students: calculate the orthonormal basis
  # Fill out function and remove
  #raise NotImplementedError("Student exercise: implement the orthonormal basis function")
  #################################################

  # Normalize vector u
  u = np.array([u[0]/np.linalg.norm(u),u[1]/np.linalg.norm(u)])

  # Calculate vector w that is orthogonal to u
  w = np.array([-u[1],u[0]])

  # Put in matrix form
  W = np.column_stack([u, w])

  return W


# Set up parameters
np.random.seed(2020)  # set random seed
variance_1 = 1
variance_2 = 1
corr_coef = 0.8
u = np.array([3, 1])

# Compute covariance matrix
cov_matrix = calculate_cov_matrix(variance_1, variance_2, corr_coef)

# Generate data
X = get_data(cov_matrix)

# Get orthonomal basis
W = define_orthonormal_basis(u)

# Visualize
plot_basis_vectors(X, W)

# %% [markdown]
# ## Change to orthonormal basis

# %% [markdown]
# Since W is orthonormal, we can project the data into our new basis using simple matrix multiplication : Y=XW
# 
# For X and Y:
# - each row = each vector
# - each col = each coordinate 

# %%
def change_of_basis(X, W):
  """
  Projects data onto new basis W.

  Args:
    X (numpy array of floats) : Data matrix each column corresponding to a
                                different random variable
    W (numpy array of floats) : new orthonormal basis columns correspond to
                                basis vectors

  Returns:
    (numpy array of floats)    : Data matrix expressed in new basis
  """

  #################################################
  ## TODO for students: project the data onto a new basis W
  # Fill out function and remove
  # raise NotImplementedError("Student exercise: implement change of basis")
  #################################################

  # Project data onto new basis described by W
  Y = X @ W

  return Y


# Project data to new basis
Y = change_of_basis(X, W)

# Visualize
plot_data_new_basis(Y)


