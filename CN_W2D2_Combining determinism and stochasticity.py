# %% [markdown]
# **dynamical systems with stochasticity**: processes for which we know some dynamics, but there are some random aspects as well

# %% [markdown]
# ## Setup

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt

# @title Figure Settings
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import ipywidgets as widgets  # interactive display
%config InlineBackend.figure_format = 'retina'
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

# @title Plotting Functions

def plot_random_walk_sims(sims, nsims=10):
  """Helper for exercise 3A"""
  plt.figure()
  plt.plot(sim[:nsims, :].T)
  plt.xlabel('time')
  plt.ylabel('position x')
  plt.show()


def plot_mean_var_by_timestep(mu, var):
  """Helper function for exercise 3A.2"""
  fig, (ah1, ah2) = plt.subplots(2)

  # plot mean of distribution as a function of time
  ah1.plot(mu)
  ah1.set(ylabel='mean')
  ah1.set_ylim([-5, 5])

  # plot variance of distribution as a function of time
  ah2.plot(var)
  ah2.set(xlabel='time')
  ah2.set(ylabel='variance')

  plt.show()


def plot_ddm(t, x, xinfty, lam, x0):
  plt.figure()
  plt.plot(t, xinfty * (1 - lam**t) + x0 * lam**t, 'r',
           label='deterministic solution')
  plt.plot(t, x, 'k.', label='simulation')  # simulated data pts
  plt.xlabel('t')
  plt.ylabel('x')
  plt.legend()
  plt.show()


def var_comparison_plot(empirical, analytical):
  plt.figure()
  plt.plot(empirical, analytical, '.', markersize=15)
  plt.xlabel('empirical equilibrium variance')
  plt.ylabel('analytic equilibrium variance')
  plt.plot(np.arange(8), np.arange(8), 'k', label='45 deg line')
  plt.legend()
  plt.grid(True)
  plt.show()


def plot_dynamics(x, t, lam, xinfty=0):
  """ Plot the dynamics """
  plt.figure()
  plt.title('$\lambda=%0.1f$' % lam, fontsize=16)
  x0 = x[0]
  plt.plot(t, xinfty + (x0 - xinfty) * lam**t, 'r', label='analytic solution')
  plt.plot(t, x, 'k.', label='simulation')  # simulated data pts
  plt.ylim(0, x0+1)
  plt.xlabel('t')
  plt.ylabel('x')
  plt.legend()
  plt.show()

# %% [markdown]
# ## Random Walks

# %%
# @markdown Execute to simulate and visualize a random walk
# parameters of simulation
T = 100
t = np.arange(T)
x = np.zeros_like(t)
np.random.seed(2020) # set random seed

# initial position
x[0] = 0

# step forward in time
for k in range(len(t)-1):
    # choose randomly between -1 and 1 (coin flip)
    this_step = np.random.choice([-1,1])

    # make the step
    x[k+1] = x[k] + this_step

# plot this trajectory
plt.figure()
plt.step(t, x)
plt.xlabel('time')
plt.ylabel('position x')
plt.show()

# %%
def random_walk_simulator(N, T, mu=0, sigma=1):
  '''Simulate N random walks for T time points. At each time point, the step
      is drawn from a Gaussian distribution with mean mu and standard deviation
      sigma.

  Args:
    T (integer) : Duration of simulation in time steps
    N (integer) : Number of random walks
    mu (float) : mean of step distribution
    sigma (float) : standard deviation of step distribution

  Returns:
    (numpy array) : NxT array in which each row corresponds to trajectory
  '''

  ###############################################################################
  ## TODO: Code the simulated random steps to take
  ## Hints: you can generate all the random steps in one go in an N x T matrix
  ## raise NotImplementedError('Complete random_walk_simulator_function')
  ###############################################################################
  # generate all the random steps for all steps in all simulations in one go
  # produces a N x T array
  steps = np.random.normal(mu, sigma, size=(N, T))

  # compute the cumulative sum of all the steps over the time axis
  sim = np.cumsum(steps, axis=1)

  return sim


np.random.seed(2020) # set random seed

# simulate 1000 random walks for 10000 time steps
sim = random_walk_simulator(1000, 10000,  mu=0, sigma=1)

# take a peek at the first 10 simulations
plot_random_walk_sims(sim, nsims=10)

# %%
# @markdown Execute to visualize distribution of bacteria positions
fig = plt.figure()
# look at the distribution of positions at different times
for i, t in enumerate([1000, 2500, 10000]):

  # get mean and standard deviation of distribution at time t
  mu = sim[:, t-1].mean()
  sig2 = sim[:, t-1].std()

  # make a plot label
  mytitle = '$t=${time:d} ($\mu=${mu:.2f}, $\sigma=${var:.2f})'

  # plot histogram
  plt.hist(sim[:,t-1],
           color=['blue','orange','black'][i],
           # make sure the histograms have the same bins!
           bins=np.arange(-300, 300, 20),
           # make histograms a little see-through
           alpha=0.6,
           # draw second histogram behind the first one
           zorder=3-i,
           label=mytitle.format(time=t, mu=mu, var=sig2))

  plt.xlabel('position x')

  # plot range
  plt.xlim([-500, 250])

  # add legend
  plt.legend(loc=2)

# add title
plt.title(r'Distribution of trajectory positions at time $t$')
plt.show()

# %% [markdown]
# **diffusive process:** the mean of the distribution is independent of time, but the variance and standard deviation of the distribution scale with time

# %%
# Simulate random walks
np.random.seed(2020) # set random seed
sim = random_walk_simulator(5000, 1000, mu=0, sigma=1)

##############################################################################
# TODO: Insert your code here to compute the mean and variance of trajectory positions
# at every time point:
#raise NotImplementedError("Student exercise: need to compute mean and variance")
##############################################################################

# Compute mean
mu = np.mean(sim,0)

# Compute variance
var = np.var(sim,0)

# Visualize
plot_mean_var_by_timestep(mu, var)

# %% [markdown]
# ## The Ornstein-Uhlenbeck (OU) process

# %% [markdown]
# **drift-diffusion model** or **Ornstein-Uhlenbeck (OU) process**: the model is a combination of a _drift_ term toward $x_{\infty}$ and a _diffusion_ term that walks randomly.
# 
# The discrete version of our OU process has the following form:
# 
# \begin{equation}
# x_{k+1} = x_\infty + \lambda(x_k - x_{\infty}) + \sigma \eta
# \end{equation}
# 
# where $\eta$ is sampled from a standard normal distribution ($\mu=0, \sigma=1$).

# %%
# parameters
lam = 0.9  # decay rate
T = 100  # total Time duration in steps
x0 = 4.  # initial condition of x at time 0
xinfty = 1.  # x drifts towards this value in long time

# initialize variables
t = np.arange(0, T, 1.)
x = np.zeros_like(t)
x[0] = x0

# Step through in time
for k in range(len(t)-1):
  x[k+1] = xinfty + lam * (x[k] - xinfty)

# plot x as it evolves in time
plot_dynamics(x, t, lam, xinfty)

# %%
def simulate_ddm(lam, sig, x0, xinfty, T):
  """
  Simulate the drift-diffusion model with given parameters and initial condition.
  Args:
    lam (scalar): decay rate
    sig (scalar): standard deviation of normal distribution
    x0 (scalar): initial condition (x at time 0)
    xinfty (scalar): drift towards convergence in the limit
    T (scalar): total duration of the simulation (in steps)

  Returns:
    ndarray, ndarray: `x` for all simulation steps and the time `t` at each step
  """

  # initialize variables
  t = np.arange(0, T, 1.)
  x = np.zeros_like(t)
  x[0] = x0

  # Step through in time
  for k in range(len(t)-1):
    ##############################################################################
    ## TODO: Insert your code below then remove
    #raise NotImplementedError("Student exercise: need to implement simulation")
    ##############################################################################
    # update x at time k+1 with a deterministic and a stochastic component
    # hint: the deterministic component will be like above, and
    #   the stochastic component is drawn from a scaled normal distribution
    x[k+1] = xinfty + lam*(x[k]-xinfty) + sig*np.random.normal(0,1)

  return t, x


lam = 0.9  # decay rate
sig = 0.1  # standard deviation of diffusive process
T = 500  # total Time duration in steps
x0 = 4.  # initial condition of x at time 0
xinfty = 1.  # x drifts towards this value in long time

# Plot x as it evolves in time
np.random.seed(2020)
t, x = simulate_ddm(lam, sig, x0, xinfty, T)
plot_ddm(t, x, xinfty, lam, x0)

# %% [markdown]
# ## Variance of the OU process

# %% [markdown]
# Unlike the random walk, because there's a decay process that "pulls" $x$ back towards $x_\infty$, the variance does not grow without bound with large $t$. Instead, when it gets far from $x_\infty$, the position of $x$ is restored, until an equilibrium is reached.
# 
# The equilibrium variance for our drift-diffusion system is
# 
# \begin{equation}
# \text{Var} = \frac{\sigma^2}{1 - \lambda^2}.
# \end{equation}


