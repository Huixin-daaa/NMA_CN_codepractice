# %% [markdown]
# ## Setup

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp  # numerical integration solver

# %%
# @title Plotting Functions
def plot_trajectory(system, params, initial_condition, dt=0.1, T=6,
                    figtitle=None):

  """
  Shows the solution of a linear system with two variables in 3 plots.
  The first plot shows x1 over time. The second plot shows x2 over time.
  The third plot shows x1 and x2 in a phase portrait.

  Args:
    system (function): a function f(x) that computes a derivative from
                        inputs (t, [x1, x2], *params)
    params (list or tuple): list of parameters for function "system"
    initial_condition (list or array): initial condition x0
    dt (float): time step of simulation
    T (float): end time of simulation
    figtitlte (string): title for the figure

  Returns:
    nothing, but it shows a figure
  """

  # time points for which we want to evaluate solutions
  t = np.arange(0, T, dt)

  # Integrate
  # use built-in ode solver
  solution = solve_ivp(system,
                    t_span=(0, T),
                    y0=initial_condition, t_eval=t,
                    args=(params),
                    dense_output=True)
  x = solution.y

  # make a color map to visualize time
  timecolors = np.array([(1 , 0 , 0, i)  for i in t / t[-1]])

  # make a large figure
  fig, (ah1, ah2, ah3) = plt.subplots(1, 3)
  fig.set_size_inches(10, 3)

  # plot x1 as a function of time
  ah1.scatter(t, x[0,], color=timecolors)
  ah1.set_xlabel('time')
  ah1.set_ylabel('x1', labelpad=-5)

  # plot x2 as a function of time
  ah2.scatter(t, x[1], color=timecolors)
  ah2.set_xlabel('time')
  ah2.set_ylabel('x2', labelpad=-5)

  # plot x1 and x2 in a phase portrait
  ah3.scatter(x[0,], x[1,], color=timecolors)
  ah3.set_xlabel('x1')
  ah3.set_ylabel('x2', labelpad=-5)
  #include initial condition is a blue cross
  ah3.plot(x[0,0], x[1,0], 'bx')

  # adjust spacing between subplots
  plt.subplots_adjust(wspace=0.5)

  # add figure title
  if figtitle is not None:
    fig.suptitle(figtitle, size=16)
  plt.show()


def plot_streamplot(A, ax, figtitle=None, show=True):
  """
  Show a stream plot for a linear ordinary differential equation with
  state vector x=[x1,x2] in axis ax.

  Args:
    A (numpy array): 2x2 matrix specifying the dynamical system
    ax (matplotlib.axes): axis to plot
    figtitle (string): title for the figure
    show (boolean): enable plt.show()

  Returns:
    nothing, but shows a figure
  """

  # sample 20 x 20 grid uniformly to get x1 and x2
  grid = np.arange(-20, 21, 1)
  x1, x2 = np.meshgrid(grid, grid)

  # calculate x1dot and x2dot at each grid point
  x1dot = A[0,0] * x1 + A[0,1] * x2
  x2dot = A[1,0] * x1 + A[1,1] * x2

  # make a colormap
  magnitude = np.sqrt(x1dot ** 2 + x2dot ** 2)
  color = 2 * np.log1p(magnitude) #Avoid taking log of zero

  # plot
  plt.sca(ax)
  plt.streamplot(x1, x2, x1dot, x2dot, color=color,
                 linewidth=1, cmap=plt.cm.cividis,
                 density=2, arrowstyle='->', arrowsize=1.5)
  plt.xlabel(r'$x1$')
  plt.ylabel(r'$x2$')

  # figure title
  if figtitle is not None:
    plt.title(figtitle, size=16)

  # include eigenvectors
  if True:
    # get eigenvalues and eigenvectors of A
    lam, v = np.linalg.eig(A)

    # get eigenvectors of A
    eigenvector1 = v[:,0].real
    eigenvector2 = v[:,1].real

    # plot eigenvectors
    plt.arrow(0, 0, 20*eigenvector1[0], 20*eigenvector1[1],
              width=0.5, color='r', head_width=2,
              length_includes_head=True)
    plt.arrow(0, 0, 20*eigenvector2[0], 20*eigenvector2[1],
              width=0.5, color='b', head_width=2,
              length_includes_head=True)
  if show:
    plt.show()


def plot_specific_example_stream_plots(A_options):
  """
  Show a stream plot for each A in A_options

  Args:
    A (list): a list of numpy arrays (each element is A)

  Returns:
    nothing, but shows a figure
  """
  # get stream plots for the four different systems
  plt.figure(figsize=(10, 10))

  for i, A in enumerate(A_options):

    ax = plt.subplot(2, 2, 1+i)
    # get eigenvalues and eigenvectors
    lam, v = np.linalg.eig(A)

    # plot eigenvalues as title
    # (two spaces looks better than one)
    eigstr = ",  ".join([f"{x:.2f}" for x in lam])
    figtitle =f"A with eigenvalues\n"+ '[' + eigstr + ']'
    plot_streamplot(A, ax, figtitle=figtitle, show=False)

    # Remove y_labels on righthand plots
    if i % 2:
      ax.set_ylabel(None)
    if i < 2:
      ax.set_xlabel(None)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
  plt.show()

# %%
# @title Figure settings
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import ipywidgets as widgets  # interactive display
%config InlineBackend.figure_format = 'retina'
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

# %% [markdown]
# ## One-dimensional Differential Equations

# %% [markdown]
# Differential equations are equations that express the **rate of change** of the state variable $x$. One typically describes this rate of change using the derivative of $x$ with respect to time ($dx/dt$) on the left hand side of the differential equation:
# 
# \begin{equation}
# \frac{dx}{dt} = f(x)
# \end{equation}
# 
# A common notational short-hand is to write $\dot{x}$ for $\frac{dx}{dt}$.

# %% [markdown]
# ### Linear Dynamical Systems

# %% [markdown]
# A one-dimensional differential equation in $x$ of the form
# 
# $$\dot{x} = a x$$
# 
# where $a$ is a scalar.
# 
# Solutions for how $x$ evolves in time when its dynamics are governed by such a differential equation take the form
# 
# \begin{equation}
# x(t) = x_0\exp(a t)
# \end{equation}
# 
# where $x_0$ is the **initial condition** of the equation -- that is, the value of $x$ at time $0$.

# %% [markdown]
# ### Forward Euler Integration

# %% [markdown]
# \begin{equation}
# x(t_i)=x(t_{i-1})+\dot x(t_{i-1}) dt
# \end{equation}
# 
# This very simple integration scheme, known as **forward Euler integration**, works well if $dt$ is small and the ordinary differential equation is simple.

# %%
def integrate_exponential(a, x0, dt, T):
  """Compute solution of the differential equation xdot=a*x with
  initial condition x0 for a duration T. Use time step dt for numerical
  solution.

  Args:
    a (scalar): parameter of xdot (xdot=a*x)
    x0 (scalar): initial condition (x at time 0)
    dt (scalar): timestep of the simulation
    T (scalar): total duration of the simulation

  Returns:
    ndarray, ndarray: `x` for all simulation steps and the time `t` at each step
  """

  # Initialize variables
  t = np.arange(0, T, dt)
  x = np.zeros_like(t, dtype=complex)
  x[0] = x0 # This is x at time t_0

  # Step through system and integrate in time
  for k in range(1, len(t)):

    ###################################################################
    ## Fill out the following then remove
    # raise NotImplementedError("Student exercise: need to implement simulation")
    ###################################################################

    # for each point in time, compute xdot from x[k-1]
    xdot = a * x[k-1]

    # Update x based on x[k-1] and xdot
    x[k] = x[k-1]+xdot*dt

  return x, t


# Choose parameters
a = -0.5    # parameter in f(x)
T = 10      # total Time duration
dt = 0.001  # timestep of our simulation
x0 = 1.     # initial condition of x at time 0

# Use Euler's method
x, t = integrate_exponential(a, x0, dt, T)

# Visualize
plt.plot(t, x.real)
plt.xlabel('Time (s)')
plt.ylabel('x')
plt.show()

# %% [markdown]
# ## Oscillatory Dynamics

# %% [markdown]
# The behavior of a one-dimensional linear dynamical system $\dot{x} = a x$ is determined by $a$, which may be a complex valued number. Knowing $a$, we know about the stability and oscillatory dynamics of the system.

# %%
# @markdown Make sure you execute this cell to enable the widget!

# parameters
T = 5  # total Time duration
dt = 0.0001  # timestep of our simulation
x0 = 1.  # initial condition of x at time 0

@widgets.interact
def plot_euler_integration(real=(-2, 2, .2), imaginary=(-4, 7, .1)):

  a = complex(real, imaginary)
  x, t = integrate_exponential(a, x0, dt, T)
  plt.plot(t, x.real)  # integrate exponential returns complex
  plt.grid(True)
  plt.xlabel('Time (s)')
  plt.ylabel('x')
  plt.show()

# %% [markdown]
# ## Deterministic Linear Dynamics in Two Dimensions

# %% [markdown]
# Adding one additional variable (or _dimension_) adds more variety of behaviors
# 
# $$\dot{x}_1 = {a}_{11} x_1 + {a}_{12} x_2$$
# $$\dot{x}_2 = {a}_{21} x_1 + {a}_{22} x_2$$ 
# 
# 
# We can write the two equations that describe our system as one (vector-valued) linear ordinary differential equation:
# 
# $$\dot{\mathbf{x}} = \mathbf{A} \mathbf{x}$$

# %%
def system(t, x, a00, a01, a10, a11):
  '''
  Compute the derivative of the state x at time t for a linear
  differential equation with A matrix [[a00, a01], [a10, a11]].

  Args:
    t (float): time
    x (ndarray): state variable
    a00, a01, a10, a11 (float): parameters of the system

  Returns:
    ndarray: derivative xdot of state variable x at time t
  '''
  #################################################
  ## TODO for students: Compute xdot1 and xdot2 ##
  ## Fill out the following then remove
  # raise NotImplementedError("Student exercise: say what they should have done")
  #################################################

  # compute x1dot and x2dot
  x1dot = a00*x[0]+a01*x[1]
  x2dot = a10*x[0]+a11*x[1]

  return np.array([x1dot, x2dot])


# Set parameters
T = 6 # total time duration
dt = 0.1 # timestep of our simulation
A = np.array([[2, -5],
              [1, -2]])
x0 = [-0.1, 0.2]

# Simulate and plot trajectories
plot_trajectory(system, [A[0, 0], A[0, 1], A[1, 0], A[1, 1]], x0, dt=dt, T=T)

# %% [markdown]
# ## Stream Plots

# %% [markdown]
# We can think of a initial condition ${\bf x}_0=(x_{1_0},x_{2_0})$  as coordinates for a position in a space. For a 2x2 matrix $\bf A$, a stream plot computes at each position $\bf x$ a small arrow that indicates $\bf Ax$ and then connects the small arrows to form _stream lines_. Remember from the beginning of this tutorial that $\dot {\bf x} = \bf Ax$ is the rate of change of $\bf x$. So the stream lines indicate how a system changes. If you are interested in a particular initial condition ${\bf x}_0$, just find the corresponding position in the stream plot. The stream line that goes through that point in the stream plot indicates ${\bf x}(t)$.

# %% [markdown]
# ### Interpreting Eigenvalues and Eigenvectors

# %%
# @markdown Execute this cell to see stream plots

A_option_1 = np.array([[2, -5], [1, -2]])
A_option_2 = np.array([[3,4], [1, 2]])
A_option_3 = np.array([[-1, -1], [0, -0.25]])
A_option_4 = np.array([[3, -2], [2, -2]])

A_options = [A_option_1, A_option_2, A_option_3, A_option_4]
plot_specific_example_stream_plots(A_options)

# %% [markdown]
# In top-left A, both eigenvalues are imaginary (no real component, the two
# eigenvalues are complex conjugate pairs), so the solutions are all stable
# oscillations. The eigenvectors are also complex conjugate pairs (that's why
# we see them plotted on top of each other). They point in the direction of the
# major axis of the elliptical trajectories.
# 
# In the top-right A, both eigenvalues are positive, so they are growing. The larger
# eigenvalue direction (red) grows faster than the other direction (blue),
# so trajectories all eventually follow the red eigenvector direction. Those that
# start close to the blue direction follow blue for a bit initially.
# 
# In the bottom-left A, both eigenvalues are negative, so they are both decaying.
# All solutions decay towards the origin [0, 0]. The red eigenvalue is larger in
# magnitude, so decay is faster along the red eigenvector.
# 
# In the bottom-right A, one eigenvalue is positive (red) and one eigenvalue is negative
# (blue). This makes the shape of the landscape the shape of a saddle (named after
# the saddle that one puts on a horse for a rider). Trajectories decay along the
# blue eigenvector but grows along the red eigenvector.

# %% [markdown]
# 


