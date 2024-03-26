# %% [markdown]
# Here we will assume that the world state is _binary_ ($\pm 1$) and _constant_ over time, but allow for multiple observations over time. 
# - We will use the *Sequential Probability Ratio Test* (SPRT) to infer which state is true. 
# - This leads to the *Drift Diffusion Model (DDM)* where evidence accumulates until reaching a stopping criterion.

# %% [markdown]
# ## Setup

# %%
# Imports
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.special import erf

# @title Figure Settings
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import ipywidgets as widgets  # interactive display
%config InlineBackend.figure_format = 'retina'
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/NMA2020/nma.mplstyle")

# @title Plotting functions

def plot_accuracy_vs_stoptime(mu, sigma, stop_time_list, accuracy_analytical_list, accuracy_list=None):
  """Simulate and plot a SPRT for a fixed amount of times given a std.

  Args:
    mu (float): absolute mean value of the symmetric observation distributions
    sigma (float): Standard deviation of the observations.
    stop_time_list (int): List of number of steps to run before stopping.
    accuracy_analytical_list (int): List of analytical accuracies for each stop time
    accuracy_list (int (optional)): List of simulated accuracies for each stop time
  """
  T = stop_time_list[-1]
  fig, ax = plt.subplots(figsize=(12,8))
  ax.set_xlabel('Stop Time')
  ax.set_ylabel('Average Accuracy')
  ax.plot(stop_time_list, accuracy_analytical_list)
  if accuracy_list is not None:
    ax.plot(stop_time_list, accuracy_list)
  ax.legend(['analytical','simulated'], loc='upper center')

  # Show two gaussian
  stop_time_list_plot = [max(1,T//10), T*2//3]
  sigma_st_max = 2*mu*np.sqrt(stop_time_list_plot[-1])/sigma
  domain = np.linspace(-3*sigma_st_max,3*sigma_st_max,50)
  for stop_time in stop_time_list_plot:
    ins = ax.inset_axes([stop_time/T,0.05,0.2,0.3])
    for pos in ['right', 'top', 'bottom', 'left']:
      ins.spines[pos].set_visible(False)
    ins.axis('off')
    ins.set_title(f"stop_time={stop_time}")

    left = np.zeros_like(domain)
    mu_st = 4*mu*mu*stop_time/2/sigma**2
    sigma_st = 2*mu*np.sqrt(stop_time)/sigma
    for i, mu1 in enumerate([-mu_st,mu_st]):
      rv = stats.norm(mu1, sigma_st)
      offset = rv.pdf(domain)
      lbl = "summed evidence" if i == 1 else ""
      color = "crimson"
      ls = "solid" if i==1 else "dashed"
      ins.plot(domain, left+offset, label=lbl, color=color,ls=ls)

    rv = stats.norm(mu_st, sigma_st)
    domain0 = np.linspace(-3*sigma_st_max,0,50)
    offset = rv.pdf(domain0)
    ins.fill_between(domain0, np.zeros_like(domain0), offset, color="crimson", label="error")
    ins.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

  plt.show(fig)

  # @title Helper Functions

def simulate_and_plot_SPRT_fixedtime(mu, sigma, stop_time, num_sample,
                                     verbose=True):
  """Simulate and plot a SPRT for a fixed amount of time given a std.

  Args:
    mu (float): absolute mean value of the symmetric observation distributions
    sigma (float): Standard deviation of the observations.
    stop_time (int): Number of steps to run before stopping.
    num_sample (int): The number of samples to plot.
    """

  evidence_history_list = []
  if verbose:
    print("Trial\tTotal_Evidence\tDecision")
  for i in range(num_sample):
    evidence_history, decision, Mvec = simulate_SPRT_fixedtime(mu, sigma, stop_time)
    if verbose:
      print("{}\t{:f}\t{}".format(i, evidence_history[-1], decision))
    evidence_history_list.append(evidence_history)

  fig, ax = plt.subplots()
  maxlen_evidence = np.max(list(map(len,evidence_history_list)))
  ax.plot(np.zeros(maxlen_evidence), '--', c='red', alpha=1.0)
  for evidences in evidence_history_list:
    ax.plot(np.arange(len(evidences)), evidences)
    ax.set_xlabel("Time")
    ax.set_ylabel("Accumulated log likelihood ratio")
    ax.set_title("Log likelihood ratio trajectories under the fixed-time " +
                  "stopping rule")

  plt.show(fig)


def simulate_and_plot_SPRT_fixedthreshold(mu, sigma, num_sample, alpha,
                                          verbose=True):
  """Simulate and plot a SPRT for a fixed amount of times given a std.

  Args:
    mu (float): absolute mean value of the symmetric observation distributions
    sigma (float): Standard deviation of the observations.
    num_sample (int): The number of samples to plot.
    alpha (float): Threshold for making a decision.
  """
  # calculate evidence threshold from error rate
  threshold = threshold_from_errorrate(alpha)

  # run simulation
  evidence_history_list = []
  if verbose:
    print("Trial\tTime\tAccumulated Evidence\tDecision")
  for i in range(num_sample):
    evidence_history, decision, Mvec = simulate_SPRT_threshold(mu, sigma, threshold)
    if verbose:
      print("{}\t{}\t{:f}\t{}".format(i, len(Mvec), evidence_history[-1],
                                      decision))
    evidence_history_list.append(evidence_history)

  fig, ax = plt.subplots()
  maxlen_evidence = np.max(list(map(len,evidence_history_list)))
  ax.plot(np.repeat(threshold,maxlen_evidence + 1), c="red")
  ax.plot(-np.repeat(threshold,maxlen_evidence + 1), c="red")
  ax.plot(np.zeros(maxlen_evidence + 1), '--', c='red', alpha=0.5)

  for evidences in evidence_history_list:
      ax.plot(np.arange(len(evidences) + 1), np.concatenate([[0], evidences]))

  ax.set_xlabel("Time")
  ax.set_ylabel("Accumulated log likelihood ratio")
  ax.set_title("Log likelihood ratio trajectories under the threshold rule")

  plt.show(fig)


def simulate_and_plot_accuracy_vs_threshold(mu, sigma, threshold_list, num_sample):
  """Simulate and plot a SPRT for a set of thresholds given a std.

  Args:
    mu (float): absolute mean value of the symmetric observation distributions
    sigma (float): Standard deviation of the observations.
    alpha_list (float): List of thresholds for making a decision.
    num_sample (int): The number of samples to plot.
  """
  accuracies, decision_speeds = simulate_accuracy_vs_threshold(mu, sigma,
                                                               threshold_list,
                                                               num_sample)

  # Plotting
  fig, ax = plt.subplots()
  ax.plot(decision_speeds, accuracies, linestyle="--", marker="o")
  ax.plot([np.amin(decision_speeds), np.amax(decision_speeds)],
          [0.5, 0.5], c='red')
  ax.set_xlabel("Average Decision speed")
  ax.set_ylabel('Average Accuracy')
  ax.set_title("Speed/Accuracy Tradeoff")
  ax.set_ylim(0.45, 1.05)

  plt.show(fig)


def threshold_from_errorrate(alpha):
  """Calculate log likelihood ratio threshold from desired error rate `alpha`

  Args:
    alpha (float): in (0,1), the desired error rate

  Return:
    threshold: corresponding evidence threshold
  """
  threshold = np.log((1. - alpha) / alpha)
  return threshold

# %% [markdown]
# ## Sequential Probability Ratio Test as a Drift Diffusion Model

# %% [markdown]
# ### Sequential Probability Ratio Test
# 
# The Sequential Probability Ratio Test is a likelihood ratio test for determining which of two hypotheses is more likely. It is appropriate for sequential independent and identially distributed (iid) data. iid means that the data comes from the same distribution.
# 
# Assume we take a series of measurements, from time 1 up to time t ($m_{1:t}$), and that our state is either +1 or -1. We want to figure out what the state is, given our measurements. To do this, we can compare the total evidence up to time $t$ for our two hypotheses (that the state is +1 or that the state is -1). We do this by computing a likelihood ratio: the ratio of the likelihood of all these measurements given the state is +1, $p(m_{1:t}|s=+1)$, to the likelihood of the measurements given the state is -1, $p(m_{1:t}|s=-1)$. This is our likelihood ratio test. In fact, we want to take the log of this likelihood ratio to give us the log likelihood ratio $L_T$.
# 
# \begin{align*}
# L_T &= log\frac{p(m_{1:t}|s=+1)}{p(m_{1:t}|s=-1)}
# \end{align*}
# 
# Since our data is independent and identically distribution, the probability of all measurements given the state equals the product of the separate probabilities of each measurement given the state ($p(m_{1:t}|s) = \prod_{t=1}^T p(m_t | s) $). We can substitute this in and use log properties to convert to a sum.
# 
# \begin{align*}
# L_T &= log\frac{p(m_{1:t}|s=+1)}{p(m_{1:t}|s=-1)}\\
# &= log\frac{\prod_{t=1}^Tp(m_{t}|s=+1)}{\prod_{t=1}^Tp(m_{t}|s=-1)}\\
# &= \sum_{t=1}^T log\frac{p(m_{t}|s=+1)}{p(m_{t}|s=-1)}\\
# &= \sum_{t=1}^T \Delta_t
# \end{align*}
# 
# In the last line, we have used $\Delta_t = log\frac{p(m_{t}|s=+1)}{p(m_{t}|s=-1)}$.
# 
# To get the full log likelihood ratio, we are summing up the log likelihood ratios at each time step. The log likelihood ratio at a time step ($L_T$) will equal the ratio at the previous time step ($L_{T-1}$) plus the ratio for the measurement at that time step, given by $\Delta_T$:
# 
# \begin{align*}
# L_T =  L_{T-1} + \Delta_T
# \end{align*}
# 
# The SPRT states that if $L_T$ is positive, then the state $s=+1$ is more likely than $s=-1$!

# %% [markdown]
# ### Sequential Probability Ratio Test as a Drift Diffusion Model
# 
# Let's assume that the probability of seeing a measurement given the state is a Gaussian (Normal) distribution where the mean ($\mu$) is different for the two states but the standard deviation ($\sigma$) is the same:
# 
# \begin{align*}
# p(m_t | s = +1) &= \mathcal{N}(\mu, \sigma^2)\\
# p(m_t | s = -1) &= \mathcal{N}(-\mu, \sigma^2)\\
# \end{align*}
# 
# We can write the new evidence (the log likelihood ratio for the measurement at time $t$) as
# 
# $$\Delta_t=b+c\epsilon_t$$
# 
# The first term, $b$, is a consistant value and equals $b=2\mu^2/\sigma^2$. This term favors the actual hidden state. The second term, $c\epsilon_t$ where $\epsilon_t\sim\mathcal{N}(0,1)$, is a standard random variable which is scaled by the diffusion $c=2\mu/\sigma$. 
# 
# The accumulation of evidence will thus "drift" toward one outcome, while "diffusing" in random directions, hence the term **"drift-diffusion model" (DDM)**. The process is most likely (but not guaranteed) to reach the correct outcome eventually.

# %% [markdown]
# #### derivation

# %% [markdown]
# Assume measurements are Gaussian-distributed with different means depending on the discrete latent variable $s$:
# 
# \begin{equation}
# p(m|s=\pm 1) = \mathcal{N}\left(\mu_\pm,\sigma^2\right)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(m-\mu_\pm)^2}{2\sigma^2}\right]}
# \end{equation}
# 
# In the log likelihood ratio for a single data point $m_i$, the normalizations cancel to give
# 
# \begin{equation}
# \Delta_t=\log \frac{p(m_t|s=+1)}{p(m_t|s=-1)} = \frac{1}{2\sigma^2}\left[-\left(m_t-\mu_+\right)^2 + (m_t-\mu_-)^2\right] \tag{5}
# \end{equation}
# 
# It's convenient to rewrite $m=\mu_\pm + \sigma \epsilon$, where $\epsilon\sim \mathcal{N}(0,1)$ is a standard Gaussian variable with zero mean and unit variance. (Why does this give the correct probability for $m$?). The preceding formula can then be rewritten as
# $$\Delta_t = \frac{1}{2\sigma^2}\left( -((\mu_\pm+\sigma\epsilon)-\mu_+)^2 + ((\mu_\pm+\sigma\epsilon)-\mu_-)^2\right) \tag{5}$$
# Let's assume that $s=+1$ so $\mu_\pm=\mu_+$ (if $s=-1$ then the result is the same with a reversed sign). In that case, the means in the first term $m_t-\mu_+$ cancel, leaving
# 
# \begin{equation}
# \Delta_t = \frac{\delta^2\mu^2}{2\sigma^2}+\frac{\delta\mu}{\sigma}\epsilon_t \tag{5}
# \end{equation}
# 
# where $\delta\mu=\mu_+-\mu_-$. If we take $\mu_\pm=\pm\mu$, then $\delta\mu=2\mu$, and
# 
# \begin{equation}
# \Delta_t=2\frac{\mu^2}{\sigma^2}+2\frac{\mu}{\sigma}\epsilon_t
# \end{equation}
# 
# The first term is a constant *drift*, and the second term is a random *diffusion*.
# 
# The SPRT says that we should add up these evidences, $L_T=\sum_{t=1}^T \Delta_t$. Note that the $\Delta_t$ are independent. Recall that for independent random variables, the mean of a sum is the sum of the means. And the variance of a sum is the sum of the variances.
# 
# </details>
# 
# Adding these $\Delta_t$ over time gives
# 
# \begin{equation}
# L_T\sim\mathcal{N}\left(2\frac{\mu^2}{\sigma^2}T,\ 4\frac{\mu^2}{\sigma^2}T\right)=\mathcal{N}(bT,c^2T)
# \end{equation}
# 
# as claimed. The log-likelihood ratio $L_t$ is a biased random walk --- normally distributed with a time-dependent mean and variance. This is the **Drift Diffusion Model**.

# %%
# @markdown Execute this cell to enable the helper function `log_likelihood_ratio`

def log_likelihood_ratio(Mvec, p0, p1):
  """Given a sequence(vector) of observed data, calculate the log of
  likelihood ratio of p1 and p0

  Args:
    Mvec (numpy vector):           A vector of scalar measurements
    p0 (Gaussian random variable): A normal random variable with `logpdf'
                                    method
    p1 (Gaussian random variable): A normal random variable with `logpdf`
                                    method

  Returns:
    llvec: a vector of log likelihood ratios for each input data point
  """
  return p1.logpdf(Mvec) - p0.logpdf(Mvec)

# %%
def simulate_SPRT_fixedtime(mu, sigma, stop_time, true_dist = 1):
  """Simulate a Sequential Probability Ratio Test with fixed time stopping
  rule. Two observation models are 1D Gaussian distributions N(1,sigma^2) and
  N(-1,sigma^2).

  Args:
    mu (float): absolute mean value of the symmetric observation distributions
    sigma (float): Standard deviation of observation models
    stop_time (int): Number of samples to take before stopping
    true_dist (1 or -1): Which state is the true state.

  Returns:
    evidence_history (numpy vector): the history of cumulated evidence given
                                      generated data
    decision (int): 1 for s = 1, -1 for s = -1
    Mvec (numpy vector): the generated sequences of measurement data in this trial
  """

  #################################################
  ## TODO for students ##
  # Fill out function and remove
  #raise NotImplementedError("Student exercise: complete simulate_SPRT_fixedtime")
  #################################################

  # Set means of observation distributions
  assert mu > 0, "Mu should be > 0"
  mu_pos = mu
  mu_neg = -mu

  # Make observation distributions
  p_pos = stats.norm(loc = mu_pos, scale = sigma)
  p_neg = stats.norm(loc = mu_neg, scale = sigma)

  # Generate a random sequence of measurements
  if true_dist == 1:
    Mvec = p_pos.rvs(size = stop_time)
  else:
    Mvec = p_neg.rvs(size = stop_time)

  # Calculate log likelihood ratio for each measurement (delta_t)
  ll_ratio_vec = log_likelihood_ratio(Mvec, p_neg, p_pos)

  # STEP 1: Calculate accumulated evidence (S) given a time series of evidence (hint: np.cumsum)
  evidence_history = np.cumsum(ll_ratio_vec)

  # STEP 2: Make decision based on the sign of the evidence at the final time.
  decision = np.sign(evidence_history[-1])

  return evidence_history, decision, Mvec


# Set random seed
np.random.seed(100)

# Set model parameters
mu = .2
sigma = 3.5  # standard deviation for p+ and p-
num_sample = 10  # number of simulations to run
stop_time = 150 # number of steps before stopping

# Simulate and visualize
simulate_and_plot_SPRT_fixedtime(mu, sigma, stop_time, num_sample)

# %% [markdown]
# 1) Higher noise, or higher sigma, means that the evidence accumulation varies up
#    and down more. You are more likely to make a wrong decision with high noise,
#    since the accumulated log likelihood ratio is more likely to be negative at the end
#    despite the true distribution being s = +1.
# 
# 2) When sigma is very small, the cumulated log likelihood ratios are basically a linear
#    diagonal line. This is because each new measurement will be very similar (since they are
#    being drawn from a Gaussian with a tiny standard deviation)
# 
# 3) You are more likely to be wrong with a small number of time steps before decision. There is
#    more change that the noise will affect the decision.

# %% [markdown]
# ## Analyzing the DDM: accuracy vs stopping time

# %%
def simulate_accuracy_vs_stoptime(mu, sigma, stop_time_list, num_sample,
                                  no_numerical=False):
  """Calculate the average decision accuracy vs. stopping time by running
  repeated SPRT simulations for each stop time.

  Args:
      mu (float): absolute mean value of the symmetric observation distributions
      sigma (float): standard deviation for observation model
      stop_list_list (list-like object): a list of stopping times to run over
      num_sample (int): number of simulations to run per stopping time
      no_numerical (bool): flag that indicates the function to return analytical values only

  Returns:
      accuracy_list: a list of average accuracies corresponding to input
                      `stop_time_list`
      decisions_list: a list of decisions made in all trials
  """

  #################################################
  ## TODO for students##
  # Fill out function and remove
  #raise NotImplementedError("Student exercise: complete simulate_accuracy_vs_stoptime")
  #################################################

  # Determine true state (1 or -1)
  true_dist = 1

  # Set up tracker of accuracy and decisions
  accuracies = np.zeros(len(stop_time_list),)
  accuracies_analytical = np.zeros(len(stop_time_list),)
  decisions_list = []

  # Loop over stop times
  for i_stop_time, stop_time in enumerate(stop_time_list):

    if not no_numerical:
      # Set up tracker of decisions for this stop time
      decisions = np.zeros((num_sample,))

      # Loop over samples
      for i in range(num_sample):

        # STEP 1: Simulate run for this stop time (hint: use output from last exercise)
        _, decision, _= simulate_SPRT_fixedtime(mu, sigma, stop_time, true_dist)

        # Log decision
        decisions[i] = decision

      # STEP 2: Calculate accuracy by averaging over trials
      accuracies[i_stop_time] = np.sum(decisions == true_dist)/len(decisions)

      # Log decision
      decisions_list.append(decisions)

    # Calculate analytical accuracy
    sigma_sum_gaussian = sigma / np.sqrt(stop_time)
    accuracies_analytical[i_stop_time] = 0.5 + 0.5 * erf(mu / np.sqrt(2) / sigma_sum_gaussian)

  return accuracies, accuracies_analytical, decisions_list


# Set random seed
np.random.seed(100)

# Set parameters of model
mu = 0.5
sigma = 4.65  # standard deviation for observation noise
num_sample = 100  # number of simulations to run for each stopping time
stop_time_list = np.arange(1, 150, 10) # Array of stopping times to use


# Calculate accuracies for each stop time
accuracies, accuracies_analytical, _ = simulate_accuracy_vs_stoptime(mu, sigma, stop_time_list,
                                                   num_sample)

# Visualize
plot_accuracy_vs_stoptime(mu, sigma, stop_time_list, accuracies_analytical, accuracies)


