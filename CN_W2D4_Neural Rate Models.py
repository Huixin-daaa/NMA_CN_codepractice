# %% [markdown]
# We will simulate and study one of the simplest models of biological neuronal networks. Instead of modeling and simulating individual excitatory neurons (e.g., LIF models that you implemented yesterday), we will treat them as a single homogeneous population and approximate their dynamics using a single one-dimensional equation describing the evolution of their average spiking rate in time.

# %% [markdown]
# **Steps:**
# - Write the equation for the firing rate dynamics of a 1D excitatory population.
# - Visualize the response of the population as a function of parameters such as threshold level and gain, using the frequency-current (F-I) curve.
# - Numerically simulate the dynamics of the excitatory population and find the fixed points of the system.

# %% [markdown]
# ## Setup

# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt  # root-finding algorithm

# @title Figure Settings
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import ipywidgets as widgets  # interactive display
%config InlineBackend.figure_format = 'retina'
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

# @title Plotting Functions

def plot_fI(x, f):
  plt.figure(figsize=(6, 4))  # plot the figure
  plt.plot(x, f, 'k')
  plt.xlabel('x (a.u.)', fontsize=14)
  plt.ylabel('F(x)', fontsize=14)
  plt.show()


def plot_dr_r(r, drdt, x_fps=None):
  plt.figure()
  plt.plot(r, drdt, 'k')
  plt.plot(r, 0. * r, 'k--')
  if x_fps is not None:
    plt.plot(x_fps, np.zeros_like(x_fps), "ko", ms=12)
  plt.xlabel(r'$r$')
  plt.ylabel(r'$\frac{dr}{dt}$', fontsize=20)
  plt.ylim(-0.1, 0.1)
  plt.show()


def plot_dFdt(x, dFdt):
  plt.figure()
  plt.plot(x, dFdt, 'r')
  plt.xlabel('x (a.u.)', fontsize=14)
  plt.ylabel('dF(x)', fontsize=14)
  plt.show()

# %% [markdown]
# ## Neuronal network dynamics

# %% [markdown]
# ### Dynamics of a single excitatory population
# 
# In this model, we are interested in how the population-averaged firing varies as a function of time and network parameters. Mathematically, we can describe the firing rate dynamic of a feed-forward network as:
# 
# \begin{equation}
# \tau \frac{dr}{dt} = -r + F(I_{\text{ext}})  \quad\qquad (1)
# \end{equation}
# 
# $r(t)$ represents the average firing rate of the excitatory population at time $t$, $\tau$ controls the timescale of the evolution of the average firing rate, $I_{\text{ext}}$ represents the external input, and the transfer function $F(\cdot)$ (which can be related to f-I curve of individual neurons described in the next sections) represents the population activation function in response to all received inputs.

# %%
# @markdown *Execute this cell to set default parameters for a single excitatory population model*


def default_pars_single(**kwargs):
  pars = {}

  # Excitatory parameters
  pars['tau'] = 1.     # Timescale of the E population [ms]
  pars['a'] = 1.2      # Gain of the E population
  pars['theta'] = 2.8  # Threshold of the E population

  # Connection strength
  pars['w'] = 0.  # E to E, we first set it to 0

  # External input
  pars['I_ext'] = 0.

  # simulation parameters
  pars['T'] = 20.       # Total duration of simulation [ms]
  pars['dt'] = .1       # Simulation time step [ms]
  pars['r_init'] = 0.2  # Initial value of E

  # External parameters if any
  pars.update(kwargs)

  # Vector of discretized time points [ms]
  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

  return pars

pars = default_pars_single()
print(pars)

# %% [markdown]
# ### F-I curves
# 
# **F-I** curve = the output spike frequency (**F**) in response to different injected currents (**I**)
# 
# The transfer function $F(\cdot)$ in Equation $1$ represents the gain of the population as a function of the total input. The gain is often modeled as a sigmoidal function, i.e., more input drive leads to a nonlinear increase in the population firing rate. The output firing rate will eventually saturate for high input values.
# 
# A sigmoidal $F(\cdot)$ is parameterized by its gain $a$ and threshold $\theta$.
# 
# $$ F(x;a,\theta) = \frac{1}{1+\text{e}^{-a(x-\theta)}} - \frac{1}{1+\text{e}^{a\theta}}  \quad(2)$$
# 
# The argument $x$ represents the input to the population. Note that the second term is chosen so that $F(0;a,\theta)=0$.
# 
# Many other transfer functions (generally monotonic) can be also used. Examples are the rectified linear function $ReLU(x)$ or the hyperbolic tangent $tanh(x)$.

# %%
def F(x, a, theta):
  """
  Population activation function.

  Args:
    x (float): the population input
    a (float): the gain of the function
    theta (float): the threshold of the function

  Returns:
    float: the population activation response F(x) for input x
  """
  #################################################
  ## TODO for students: compute f = F(x) ##
  # Fill out function and remove
  #raise NotImplementedError("Student exercise: implement the f-I function")
  #################################################

  # Define the sigmoidal transfer function f = F(x)
  f = 1/(1+np.exp(-a*(x-theta)))-1/(1+np.exp(a*theta))
  return f


# Set parameters
pars = default_pars_single()  # get default parameters
x = np.arange(0, 10, .1)      # set the range of input

# Compute transfer function
f = F(x, pars['a'], pars['theta'])

# Visualize
plot_fI(x, f)

# %%
# @markdown Make sure you execute this cell to enable the widget!

@widgets.interact(
    a=widgets.FloatSlider(1.5, min=0.3, max=3., step=0.3),
    theta=widgets.FloatSlider(3., min=2., max=4., step=0.2)
)


def interactive_plot_FI(a, theta):
  """
  Population activation function.

  Expecxts:
    a     : the gain of the function
    theta : the threshold of the function

  Returns:
    plot the F-I curve with give parameters
  """

  # set the range of input
  x = np.arange(0, 10, .1)
  plt.figure()
  plt.plot(x, F(x, a, theta), 'k')
  plt.xlabel('x (a.u.)', fontsize=14)
  plt.ylabel('F(x)', fontsize=14)
  plt.show()

# %% [markdown]
# 1)  a determines the slope (gain) of the rising phase of the F-I curve
# 
# 2) theta determines the input at which the function F(x) reaches its mid-value (0.5).
# That is, theta shifts the F-I curve along the horizontal axis.
# 
# For our neurons we are using in this tutorial:
# - a controls the gain of the neuron population
# - theta controls the threshold at which the neuron population starts to respond

# %% [markdown]
# ### Simulation scheme of E dynamics

# %%
# @markdown Execute this cell to enable the single population rate model simulator: `simulate_single`


def simulate_single(pars):
  """
  Simulate an excitatory population of neurons

  Args:
    pars   : Parameter dictionary

  Returns:
    rE     : Activity of excitatory population (array)

  Example:
    pars = default_pars_single()
    r = simulate_single(pars)
  """

  # Set parameters
  tau, a, theta = pars['tau'], pars['a'], pars['theta']
  w = pars['w']
  I_ext = pars['I_ext']
  r_init = pars['r_init']
  dt, range_t = pars['dt'], pars['range_t']
  Lt = range_t.size

  # Initialize activity
  r = np.zeros(Lt)
  r[0] = r_init
  I_ext = I_ext * np.ones(Lt)

  # Update the E activity
  for k in range(Lt - 1):
      dr = dt / tau * (-r[k] + F(w * r[k] + I_ext[k], a, theta))
      r[k+1] = r[k] + dr

  return r


help(simulate_single)

# %%
# @title

# @markdown Make sure you execute this cell to enable the widget!

# get default parameters
pars = default_pars_single(T=20.)

@widgets.interact(
    I_ext=widgets.FloatSlider(5.0, min=0.0, max=10., step=1.),
    tau=widgets.FloatSlider(3., min=1., max=5., step=0.2)
)


def Myplot_E_diffI_difftau(I_ext, tau):
  # set external input and time constant
  pars['I_ext'] = I_ext
  pars['tau'] = tau

  # simulation
  r = simulate_single(pars)

  # Analytical Solution
  r_ana = (pars['r_init']
           + (F(I_ext, pars['a'], pars['theta'])
           - pars['r_init']) * (1. - np.exp(-pars['range_t'] / pars['tau'])))

  # plot
  plt.figure()
  plt.plot(pars['range_t'], r, 'b', label=r'$r_{\mathrm{sim}}$(t)', alpha=0.5,
           zorder=1)
  plt.plot(pars['range_t'], r_ana, 'b--', lw=5, dashes=(2, 2),
           label=r'$r_{\mathrm{ana}}$(t)', zorder=2)
  plt.plot(pars['range_t'],
           F(I_ext, pars['a'], pars['theta']) * np.ones(pars['range_t'].size),
           'k--', label=r'$F(I_{\mathrm{ext}})$')
  plt.xlabel('t (ms)', fontsize=16.)
  plt.ylabel('Activity r(t)', fontsize=16.)
  plt.legend(loc='best', fontsize=14.)
  plt.show()

# %% [markdown]
# 1) Weak inputs to the neurons eventually result in the activity converging to zero.
#  Strong inputs to the neurons eventually result in the activity converging to max value
# 
# 2) The time constant tau, does not affect the final response reached but it determines
# the time the neurons take to reach to their fixed point.

# %% [markdown]
# We have numerically solved a system driven by a positive input. Yet, $r_E(t)$ either decays to zero or reaches a fixed non-zero value.
# 
# 1. Why doesn't the solution of the system "explode" in a finite time? In other words, what guarantees that $r_E$(t) stays finite?
# 2. Which parameter would you change in order to increase the maximum value of the response?
# 
# 1) As the F-I curve is bounded between zero and one, the system doesn't explode.
# The f-curve guarantees this property
# 2) One way to increase the maximum response is to change the f-I curve. For
# example, the ReLU is an unbounded function, and thus will increase the overall maximal
# response of the network.

# %% [markdown]
# ## Fixed points of the single population system

# %% [markdown]
# We can now extend our feed-forward network to a recurrent network, governed by the equation:
# 
# \begin{align}
# \tau \frac{dr}{dt} &= -r + F(w\cdot r + I_{\text{ext}})  \quad\qquad (3)
# \end{align}
# 
# where as before, $r(t)$ represents the average firing rate of the excitatory population at time $t$, $\tau$ controls the timescale of the evolution of the average firing rate, $I_{\text{ext}}$ represents the external input, and the transfer function $F(\cdot)$ represents the population activation function in response to all received inputs. Now we also have $w$ which denotes the strength (synaptic weight) of the recurrent input to the population.
# 
# At first the system output quickly changes, with time, it reaches its maximum/minimum value and does not change anymore. The value eventually reached by the system is called the **steady state** of the system, or the **fixed point**. Essentially, in the steady states the derivative with respect to time of the activity ($r$) is zero, i.e. $\displaystyle \frac{dr}{dt}=0$.
# 
# We can find that the steady state of the Equation. (1) by setting $\displaystyle{\frac{dr}{dt}=0}$ and solve for $r$:
# 
# \begin{equation}
# -r_{\text{steady}} + F(w\cdot r_{\text{steady}} + I_{\text{ext}};a,\theta) = 0 \qquad (4)
# \end{equation}
# 
# When it exists, the solution of Equation. (4) defines a **fixed point** of the dynamical system in Equation (3). Note that if $F(x)$ is nonlinear, it is not always possible to find an analytical solution, but the solution can be found via numerical simulations.

# %% [markdown]
# ### Visualization of the fixed points
# 
# When it is not possible to find the solution for Equation (3) analytically, a graphical approach can be taken. To that end, it is useful to plot $\displaystyle{\frac{dr}{dt}}$ as a function of $r$. The values of $r$ for which the plotted function crosses zero on the y axis correspond to fixed points.
# 
# Here, let us, for example, set $w=5.0$ and $I^{\text{ext}}=0.5$. From Equation (3), you can obtain
# 
# \begin{equation}
# \frac{dr}{dt} = [-r + F(w\cdot r + I^{\text{ext}})]\,/\,\tau
# \end{equation}
# 
# Then, plot the $dr/dt$ as a function of $r$, and check for the presence of fixed points.

# %%
def compute_drdt(r, I_ext, w, a, theta, tau, **other_pars):
  """Given parameters, compute dr/dt as a function of r.

  Args:
    r (1D array) : Average firing rate of the excitatory population
    I_ext, w, a, theta, tau (numbers): Simulation parameters to use
    other_pars : Other simulation parameters are unused by this function

  Returns
    drdt function for each value of r
  """
  #########################################################################
  # TODO compute drdt and disable the error
  # raise NotImplementedError("Finish the compute_drdt function")
  #########################################################################

  # Calculate drdt
  drdt = (-r + F(w * r + I_ext, a, theta)) / tau

  return drdt


# Define a vector of r values and the simulation parameters
r = np.linspace(0, 1, 1000)
pars = default_pars_single(I_ext=0.5, w=5)

# Compute dr/dt
drdt = compute_drdt(r, **pars)

# Visualize
plot_dr_r(r, drdt)

# %% [markdown]
# ### Numerical calculation of fixed points

# %%
# @markdown Execute this cell to enable the fixed point functions

def my_fp_single(r_guess, a, theta, w, I_ext, **other_pars):
  """
  Calculate the fixed point through drE/dt=0

  Args:
    r_guess  : Initial value used for scipy.optimize function
    a, theta, w, I_ext : simulation parameters

  Returns:
    x_fp    : value of fixed point
  """
  # define the right hand of E dynamics
  def my_WCr(x):
    r = x
    drdt = (-r + F(w * r + I_ext, a, theta))
    y = np.array(drdt)

    return y

  x0 = np.array(r_guess)
  x_fp = opt.root(my_WCr, x0).x.item()

  return x_fp


def check_fp_single(x_fp, a, theta, w, I_ext, mytol=1e-4, **other_pars):
  """
   Verify |dr/dt| < mytol

  Args:
    fp      : value of fixed point
    a, theta, w, I_ext: simulation parameters
    mytol   : tolerance, default as 10^{-4}

  Returns :
    Whether it is a correct fixed point: True/False
  """
  # calculate Equation(3)
  y = x_fp - F(w * x_fp + I_ext, a, theta)

  # Here we set tolerance as 10^{-4}
  return np.abs(y) < mytol


def my_fp_finder(pars, r_guess_vector, mytol=1e-4):
  """
  Calculate the fixed point(s) through drE/dt=0

  Args:
    pars    : Parameter dictionary
    r_guess_vector  : Initial values used for scipy.optimize function
    mytol   : tolerance for checking fixed point, default as 10^{-4}

  Returns:
    x_fps   : values of fixed points

  """
  x_fps = []
  correct_fps = []
  for r_guess in r_guess_vector:
    x_fp = my_fp_single(r_guess, **pars)
    if check_fp_single(x_fp, **pars, mytol=mytol):
      x_fps.append(x_fp)

  return x_fps


help(my_fp_finder)

# %%
# Set parameters
r = np.linspace(0, 1, 1000)
pars = default_pars_single(I_ext=0.5, w=5)

# Compute dr/dt
drdt = compute_drdt(r, **pars)

#############################################################################
# TODO for students:
# Define initial values close to the intersections of drdt and y=0
# (How many initial values? Hint: How many times do the two lines intersect?)
# Calculate the fixed point with these initial values and plot them
# raise NotImplementedError('student_exercise: find fixed points numerically')
#############################################################################

# Initial guesses for fixed points
r_guess_vector = [0.1,0.5,0.7]

# Find fixed point numerically
x_fps = my_fp_finder(pars, r_guess_vector)

# Visualize
plot_dr_r(r, drdt, x_fps)

# %%
# @markdown Make sure you execute this cell to enable the widget!

@widgets.interact(
    w=widgets.FloatSlider(4., min=1., max=7., step=0.2),
    I_ext=widgets.FloatSlider(1.5, min=0., max=3., step=0.1)
)


def plot_intersection_single(w, I_ext):
  # set your parameters
  pars = default_pars_single(w=w, I_ext=I_ext)

  # find fixed points
  r_init_vector = [0, .4, .9]
  x_fps = my_fp_finder(pars, r_init_vector)

  # plot
  r = np.linspace(0, 1., 1000)
  drdt = (-r + F(w * r + I_ext, pars['a'], pars['theta'])) / pars['tau']

  plot_dr_r(r, drdt, x_fps)

# %% [markdown]
# The fixed points of the single excitatory neuron population are determined by both
# recurrent connections w and external input I_ext. In a previous interactive demo
# we saw how the system showed two different steady-states when w = 0. But when w
# doe not equal 0, for some range of w the system shows three fixed points and the
# steady state depends on the initial conditions (i.e., r at time zero).

# %% [markdown]
# ### Relationship between trajectories & fixed points

# %%
# @markdown Execute to visualize dr/dt

def plot_intersection_single(w, I_ext):
  # set your parameters
  pars = default_pars_single(w=w, I_ext=I_ext)

  # find fixed points
  r_init_vector = [0, .4, .9]
  x_fps = my_fp_finder(pars, r_init_vector)

  # plot
  r = np.linspace(0, 1., 1000)
  drdt = (-r + F(w * r + I_ext, pars['a'], pars['theta'])) / pars['tau']

  plot_dr_r(r, drdt, x_fps)


plot_intersection_single(w = 5.0, I_ext = 0.5)

# %%
# @markdown Make sure you execute this cell to enable the widget!
pars = default_pars_single(w=5.0, I_ext=0.5)

@widgets.interact(
    r_init=widgets.FloatSlider(0.5, min=0., max=1., step=.02)
)

def plot_single_diffEinit(r_init):
  pars['r_init'] = r_init
  r = simulate_single(pars)

  plt.figure()
  plt.plot(pars['range_t'], r, 'b', zorder=1)
  plt.plot(0, r[0], 'bo', alpha=0.7, zorder=2)
  plt.xlabel('t (ms)', fontsize=16)
  plt.ylabel(r'$r(t)$', fontsize=16)
  plt.ylim(0, 1.0)
  plt.show()

# %%
# @markdown Execute this cell to see the trajectories!

pars = default_pars_single()
pars['w'] = 5.0
pars['I_ext'] = 0.5

plt.figure(figsize=(8, 5))
for ie in range(10):
  pars['r_init'] = 0.1 * ie  # set the initial value
  r = simulate_single(pars)  # run the simulation

  # plot the activity with given initial
  plt.plot(pars['range_t'], r, 'b', alpha=0.1 + 0.1 * ie,
           label=r'r$_{\mathrm{init}}$=%.1f' % (0.1 * ie))

plt.xlabel('t (ms)')
plt.title('Two steady states?')
plt.ylabel(r'$r$(t)')
plt.legend(loc=[1.01, -0.06], fontsize=14)
plt.show()

# %% [markdown]
# We have three fixed points but only two steady states showing up - what’s happening?
# 
# It turns out that the stability of the fixed points matters. If a fixed point is stable, a trajectory starting near that fixed point will stay close to that fixed point and converge to it (the steady state will equal the fixed point). If a fixed point is unstable, any trajectories starting close to it will diverge and go towards stable fixed points. In fact, the only way for a trajectory to reach a stable state at an unstable fixed point is if the initial value exactly equals the value of the fixed point.

# %%
# @markdown Execute to visualize trajectory starting at unstable fixed point
pars = default_pars_single()
pars['w'] = 5.0
pars['I_ext'] = 0.5

plt.figure(figsize=(8, 5))
for ie in range(10):
  pars['r_init'] = 0.1 * ie  # set the initial value
  r = simulate_single(pars)  # run the simulation

  # plot the activity with given initial
  plt.plot(pars['range_t'], r, 'b', alpha=0.1 + 0.1 * ie,
           label=r'r$_{\mathrm{init}}$=%.1f' % (0.1 * ie))

pars['r_init'] = x_fps[1]  # set the initial value
r = simulate_single(pars)  # run the simulation

# plot the activity with given initial
plt.plot(pars['range_t'], r, 'r', alpha=0.1 + 0.1 * ie,
          label=r'r$_{\mathrm{init}}$=%.4f' % (x_fps[1]))

plt.xlabel('t (ms)')
plt.title('Two steady states?')
plt.ylabel(r'$r$(t)')
plt.legend(loc=[1.01, -0.06], fontsize=14)
plt.show()

# %% [markdown]
# ### Inhibitory populations
# 
# Throughout the tutorial, we have assumed $w> 0 $, i.e., we considered a single population of **excitatory** neurons. What do you think will be the behavior of a population of inhibitory neurons, i.e., where $w> 0$ is replaced by $w< 0$?
# 
# The system has only one fixed point and that is at zero value. For this particular dynamics, the system will eventually converge to zero. 


