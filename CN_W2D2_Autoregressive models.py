# %% [markdown]
# ## Setup

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt

# @title Plotting Functions

def plot_residual_histogram(res):
  """Helper function for Exercise 4A"""
  plt.figure()
  plt.hist(res)
  plt.xlabel('error in linear model')
  plt.title(f'stdev of errors = {res.std():.4f}')
  plt.show()


def plot_training_fit(x1, x2, p):
  """Helper function for Exercise 4B"""
  plt.figure()
  plt.scatter(x2 + np.random.standard_normal(len(x2))*0.02,
              np.dot(x1.T, p), alpha=0.2)
  plt.title(f'Training fit, order {r} AR model')
  plt.xlabel('x')
  plt.ylabel('estimated x')
  plt.show()

  # @title Helper Functions

def ddm(T, x0, xinfty, lam, sig):
  '''
  Samples a trajectory of a drift-diffusion model.

  args:
  T (integer): length of time of the trajectory
  x0 (float): position at time 0
  xinfty (float): equilibrium position
  lam (float): process param
  sig: standard deviation of the normal distribution

  returns:
  t (numpy array of floats): time steps from 0 to T sampled every 1 unit
  x (numpy array of floats): position at every time step
  '''
  t = np.arange(0, T, 1.)
  x = np.zeros_like(t)
  x[0] = x0

  for k in range(len(t)-1):
      x[k+1] = xinfty + lam * (x[k] - xinfty) + sig * np.random.standard_normal(size=1)

  return t, x

def build_time_delay_matrices(x, r):
    """
    Builds x1 and x2 for regression

    Args:
    x (numpy array of floats): data to be auto regressed
    r (scalar): order of Autoregression model

    Returns:
    (numpy array of floats) : to predict "x2"
    (numpy array of floats) : predictors of size [r,n-r], "x1"

    """
    # construct the time-delayed data matrices for order-r AR model
    x1 = np.ones(len(x)-r)
    x1 = np.vstack((x1, x[0:-r]))
    xprime = x
    for i in range(r-1):
        xprime = np.roll(xprime, -1)
        x1 = np.vstack((x1, xprime[0:-r]))

    x2 = x[r:]

    return x1, x2


def AR_prediction(x_test, p):
    """
    Returns the prediction for test data "x_test" with the regression
    coefficients p

    Args:
    x_test (numpy array of floats): test data to be predicted
    p (numpy array of floats): regression coefficients of size [r] after
    solving the autoregression (order r) problem on train data

    Returns:
    (numpy array of floats): Predictions for test data. +1 if positive and -1
    if negative.
    """
    x1, x2 = build_time_delay_matrices(x_test, len(p)-1)

    # Evaluating the AR_model function fit returns a number.
    # We take the sign (- or +) of this number as the model's guess.
    return np.sign(np.dot(x1.T, p))

def error_rate(x_test, p):
    """
    Returns the error of the Autoregression model. Error is the number of
    mismatched predictions divided by total number of test points.

    Args:
    x_test (numpy array of floats): data to be predicted
    p (numpy array of floats): regression coefficients of size [r] after
    solving the autoregression (order r) problem on train data

    Returns:
    (float): Error (percentage).
    """
    x1, x2 = build_time_delay_matrices(x_test, len(p)-1)

    return np.count_nonzero(x2 - AR_prediction(x_test, p)) / len(x2)

# %% [markdown]
# ## Fitting data to the OU process

# %% [markdown]
# Drift-diffusion (OU) process had the following form:
# 
# \begin{equation}
# x_{k+1} = x_{\infty} + \lambda(x_k - x_{\infty}) + \sigma \eta
# \end{equation}
# 
# where $\eta$ is sampled from a standard normal distribution.
# 
# For simplicity, we set $x_\infty = 0$. Now this system takes on the form:
# 
# \begin{equation}
# x_{k+1} = \lambda x_k + \eta \,,
# \end{equation}
# 
# where $\eta$ is noise from a normal distribution.

# %%
# @markdown Execute to simulate the drift diffusion model
np.random.seed(2020) # set random seed

# parameters
T = 200
x0 = 10
xinfty = 0
lam = 0.9
sig = 0.2

# drift-diffusion model from tutorial 3
t, x = ddm(T, x0, xinfty, lam, sig)

plt.figure()
plt.title('$x_0=%d, x_{\infty}=%d, \lambda=%0.1f, \sigma=%0.1f$' % (x0, xinfty, lam, sig))
plt.plot(t, x, 'k.')
plt.xlabel('time')
plt.ylabel('position x')
plt.show()

# %% [markdown]
# * question: What if we were given these positions $x$ as they evolve in time as data, how would we get back out the dynamics of the system $\lambda$?
# * solution: Our approach is to solve for $\lambda$ as a **regression problem**.

# %%
# @markdown Execute to visualize X(k) vs. X(k+1)
# make a scatter plot of every data point in x
# at time k versus time k+1
plt.figure()
plt.scatter(x[0:-2], x[1:-1], color='k')
plt.plot([0, 10], [0, 10], 'k--', label='$x_{k+1} = x_k$ line')
plt.xlabel('$x_k$')
plt.ylabel('$x_{k+1}$')
plt.legend()
plt.show()

# %% [markdown]
# Let $\mathbf{x_1} = x_{0:T-1}$ and $\mathbf{x_2} = x_{1:T}$ be vectors of the data indexed so that they are shifted in time by one. Then, our regression problem is
# 
# \begin{equation}
# \mathbf{x}_2 = \lambda \mathbf{x}_1
# \end{equation}
# 
# * This model is **autoregressive**, where _auto_ means self. In other words, it's a **regression of the time series on itself from the past**. 
# 
# * The equation as written above is only a function of itself from _one step_ in the past, so we can call it a _first order_ autoregressive model.

# %%
# @markdown Execute to solve for lambda through autoregression
# build the two data vectors from x
x1 = x[0:-2]
x1 = x1[:, np.newaxis]**[0, 1]

x2 = x[1:-1]

# solve for an estimate of lambda as a linear regression problem
p, res, rnk, s = np.linalg.lstsq(x1, x2, rcond=None)

# here we've artificially added a vector of 1's to the x1 array,
# so that our linear regression problem has an intercept term to fit.
# we expect this coefficient to be close to 0.
# the second coefficient in the regression is the linear term:
# that's the one we're after!
lam_hat = p[1]

# plot the data points
fig = plt.figure()
plt.scatter(x[0:-2], x[1:-1], color='k')
plt.xlabel('$x_k$')
plt.ylabel('$x_{k+1}$')

# plot the 45 degree line
plt.plot([0, 10], [0, 10], 'k--', label='$x_{k+1} = x_k$ line')


# plot the regression line on top
xx = np.linspace(-sig*10, max(x), 100)
yy = p[0] + lam_hat * xx
plt.plot(xx, yy, 'r', linewidth=2, label='regression line')

mytitle = 'True $\lambda$ = {lam:.4f}, Estimate $\lambda$ = {lam_hat:.4f}'
plt.title(mytitle.format(lam=lam, lam_hat=lam_hat))
plt.legend()
plt.show()

# %%
##############################################################################
## Insert your code here take to compute the residual (error)
# raise NotImplementedError('student exercise: compute the residual error')
##############################################################################
# compute the predicted values using the autoregressive model (lam_hat), and
# the residual is the difference between x2 and the prediction

res = x2 - lam_hat * x[0:-2]

# Visualize
plot_residual_histogram(res)

# %% [markdown]
# ## Higher order autoregressive models

# %% [markdown]
# **Higher order** autoregression models a future time point based on _more than one point in the past_.
# 
# In one dimension, we can write such an order-$r$ model as
# 
# \begin{equation}
# x_{k+1} = \alpha_0 + \alpha_1 x_k + \alpha_2 x_{k-1} + \alpha_3 x_{k-2} + \dots + \alpha_{r+1} x_{k-r} \, ,
# \end{equation}
# 
# where the $\alpha$'s are the $r+1$ coefficients to be fit to the data available.

# %%
# this sequence is entirely predictable, so an AR model should work
monkey_at_typewriter = '1010101010101010101010101010101010101010101010101'

# Bonus: this sequence is also predictable, but does an order-1 AR model work?
#monkey_at_typewriter = '100100100100100100100100100100100100100'

# function to turn chars to numpy array,
# coding it this way makes the math easier
# '0' -> -1
# '1' -> +1
def char2array(s):
  m = [int(c) for c in s]
  x = np.array(m)
  return x*2 - 1


x = char2array(monkey_at_typewriter)

plt.figure()
plt.step(x, '.-')
plt.xlabel('time')
plt.ylabel('random variable')
plt.show()

# %%
# build the two data vectors from x
x1 = x[0:-2]
x1 = x1[:, np.newaxis]**[0, 1]

x2 = x[1:-1]

# solve for an estimate of lambda as a linear regression problem
p, res, rnk, s = np.linalg.lstsq(x1, x2, rcond=None)

# %%
# take a look at the resulting regression coefficients
print(f'alpha_0 = {p[0]:.2f}, alpha_1 = {p[1]:.2f}')

# %%
# data generated by 9-yr-old JAB:
# we will be using this sequence to train the data
monkey_at_typewriter = '10010101001101000111001010110001100101000101101001010010101010001101101001101000011110100011011010010011001101000011101001110000011111011101000011110000111101001010101000111100000011111000001010100110101001011010010100101101000110010001100011100011100011100010110010111000101'

# we will be using this sequence to test the data
test_monkey = '00100101100001101001100111100101011100101011101001010101000010110101001010100011110'

x = char2array(monkey_at_typewriter)
test = char2array(test_monkey)

## testing: machine generated randint should be entirely unpredictable
## uncomment the lines below to try random numbers instead
# np.random.seed(2020) # set random seed
# x = char2array(np.random.randint(2, size=500))
# test = char2array(np.random.randint(2, size=500))

plt.figure()
plt.step(x, '.-')
plt.xlabel('time')
plt.ylabel('random variable')
plt.show()

# %%
# @markdown Execute this cell to get helper function `AR_model`

def AR_model(x, r):
    """
    Solves Autoregression problem of order (r) for x

    Args:
    x (numpy array of floats): data to be auto regressed
    r (scalar): order of Autoregression model

    Returns:
    (numpy array of floats) : to predict "x2"
    (numpy array of floats) : predictors of size [r,n-r], "x1"
    (numpy array of floats): coefficients of length [r] for prediction after
    solving the regression problem "p"
    """
    x1, x2 = build_time_delay_matrices(x, r)

    # solve for an estimate of lambda as a linear regression problem
    p, res, rnk, s = np.linalg.lstsq(x1.T, x2, rcond=None)

    return x1, x2, p

help(AR_model)

# %%
##############################################################################
## TODO: Insert your code here for fitting the AR model
# raise NotImplementedError('student exercise: fit AR model')
##############################################################################
# define the model order, and use AR_model() to generate the model and prediction
r = 5
x1, x2, p = AR_model(x,r)

# Plot the Training data fit
# Note that this adds a small amount of jitter to horizontal axis for visualization purposes
plot_training_fit(x1, x2, p)

# %%
# @markdown Execute to see model performance on test data

x1_test, x2_test = build_time_delay_matrices(test, r)
plt.figure()
plt.scatter(x2_test+np.random.standard_normal(len(x2_test))*0.02,
            np.dot(x1_test.T, p), alpha=0.5)
mytitle = f'Testing fit, order {r} AR model, err = {error_rate(test, p):.3f}'
plt.title(mytitle)
plt.xlabel('test x')
plt.ylabel('estimated x')
plt.show()

# %%
# @markdown Execute to visualize errors for different orders

# range of r's to try
r = np.arange(1, 21)
err = np.ones_like(r) * 1.0

for i, rr in enumerate(r):
  # fitting the model on training data
  x1, x2, p = AR_model(x, rr)
  # computing and storing the test error
  test_error = error_rate(test, p)
  err[i] = test_error


plt.figure()
plt.plot(r, err, '.-')
plt.plot([1, r[-1]], [0.5, 0.5], 'r--', label='random chance')
plt.xlabel('Order r of AR model')
plt.ylabel('Test error')
plt.xticks(np.arange(0,25,5))
plt.legend()
plt.show()

# %%



