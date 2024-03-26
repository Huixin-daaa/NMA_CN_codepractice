# %% [markdown]
# **static synapses:** weight is always fixed
# 
# **dynamic synapses:** synaptic strength is dependent on the recent spike history; synapses can either progressively increase or decrease the size of their effects on the post-synaptic neuron, based on the recent firing rate of its presynaptic partners. 
# - This feature of synapses in the brain is called **Short-Term Plasticity** and causes synapses to undergo *Facilitation* or *Depression*.

# %% [markdown]
# ## Setup

# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np

# @title Figure Settings
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


import ipywidgets as widgets  # interactive display
%config InlineBackend.figure_format='retina'
# use NMA plot style
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")
my_layout = widgets.Layout()


# @title Plotting Functions

def my_illus_LIFSYN(pars, v_fmp, v):
  """
  Illustartion of FMP and membrane voltage

  Args:
    pars  : parameters dictionary
    v_fmp : free membrane potential, mV
    v     : membrane voltage, mV

  Returns:
    plot of membrane voltage and FMP, alongside with the spiking threshold
    and the mean FMP (dashed lines)
  """

  plt.figure(figsize=(14, 5))
  plt.plot(pars['range_t'], v_fmp, 'r', lw=1.,
           label='Free mem. pot.', zorder=2)
  plt.plot(pars['range_t'], v, 'b', lw=1.,
           label='True mem. pot', zorder=1, alpha=0.7)
  plt.axhline(pars['V_th'], 0, 1, color='k', lw=2., ls='--',
              label='Spike Threshold', zorder=1)
  plt.axhline(np.mean(v_fmp), 0, 1, color='r', lw=2., ls='--',
              label='Mean Free Mem. Pot.', zorder=1)
  plt.xlabel('Time (ms)')
  plt.ylabel('V (mV)')
  plt.legend(loc=[1.02, 0.68])
  plt.show()


def plot_volt_trace(pars, v, sp, show=True):
  """
  Plot trajetory of membrane potential for a single neuron

  Args:
    pars   : parameter dictionary
    v      : volt trajetory
    sp     : spike train

  Returns:
    figure of the membrane potential trajetory for a single neuron
  """

  V_th = pars['V_th']
  dt = pars['dt']
  if sp.size:
    sp_num = (sp/dt).astype(int) - 1
    v[sp_num] += 10

  plt.plot(pars['range_t'], v, 'b')
  plt.axhline(V_th, 0, 1, color='k', ls='--', lw=1.)
  plt.xlabel('Time (ms)')
  plt.ylabel('V (mV)')
  if show:
    plt.show()


def my_illus_STD(Poisson=False, rate=20., U0=0.5,
                 tau_d=100., tau_f=50., plot_out=True):
  """
   Only for one presynaptic train

  Args:
    Poisson    : Poisson or regular input spiking trains
    rate       : Rate of input spikes, Hz
    U0         : synaptic release probability at rest
    tau_d      : synaptic depression time constant of x [ms]
    tau_f      : synaptic facilitation time constantr of u [ms]
    plot_out   : whether ot not to plot, True or False

  Returns:
    Nothing.
  """

  T_simu = 10.0 * 1000 / (1.0 * rate)  # 10 spikes in the time window
  pars = default_pars(T=T_simu)
  dt = pars['dt']

  if Poisson:
    # Poisson type spike train
    pre_spike_train = Poisson_generator(pars, rate, n=1)
    pre_spike_train = pre_spike_train.sum(axis=0)
  else:
    # Regular firing rate
    isi_num = int((1e3/rate)/dt)  # number of dt
    pre_spike_train = np.zeros(len(pars['range_t']))
    pre_spike_train[::isi_num] = 1.

  u, R, g = dynamic_syn(g_bar=1.2, tau_syn=5., U0=U0,
                        tau_d=tau_d, tau_f=tau_f,
                        pre_spike_train=pre_spike_train,
                        dt=pars['dt'])

  if plot_out:
    plt.figure(figsize=(12, 6))

    plt.subplot(221)
    plt.plot(pars['range_t'], R, 'b', label='R')
    plt.plot(pars['range_t'], u, 'r', label='u')
    plt.legend(loc='best')
    plt.xlim((0, pars['T']))
    plt.ylabel(r'$R$ or $u$ (a.u)')
    plt.subplot(223)
    spT = pre_spike_train > 0
    t_sp = pars['range_t'][spT]  #spike times
    plt.plot(t_sp, 0. * np.ones(len(t_sp)), 'k|', ms=18, markeredgewidth=2)
    plt.xlabel('Time (ms)');
    plt.xlim((0, pars['T']))
    plt.yticks([])
    plt.title('Presynaptic spikes')

    plt.subplot(122)
    plt.plot(pars['range_t'], g, 'r', label='STP synapse')
    plt.xlabel('Time (ms)')
    plt.ylabel('g (nS)')
    plt.xlim((0, pars['T']))

    plt.tight_layout()
    plt.show()

  if not Poisson:
    return g[isi_num], g[9*isi_num]
  

  # @title Helper Functions

def my_GWN(pars, mu, sig, myseed=False):
  """
  Args:
    pars       : parameter dictionary
    mu         : noise baseline (mean)
    sig        : noise amplitute (standard deviation)
    myseed     : random seed. int or boolean
                  the same seed will give the same random number sequence

  Returns:
    I          : Gaussian White Noise (GWN) input
  """

  # Retrieve simulation parameters
  dt, range_t = pars['dt'], pars['range_t']
  Lt = range_t.size

  # set random seed
  # you can fix the seed of the random number generator so that the results
  # are reliable. However, when you want to generate multiple realizations
  # make sure that you change the seed for each new realization.
  if myseed:
      np.random.seed(seed=myseed)
  else:
      np.random.seed()

  # generate GWN
  # we divide here by 1000 to convert units to seconds.
  I_GWN = mu + sig * np.random.randn(Lt) / np.sqrt(dt / 1000.)

  return I_GWN


def Poisson_generator(pars, rate, n, myseed=False):
  """
  Generates poisson trains

  Args:
    pars       : parameter dictionary
    rate       : noise amplitute [Hz]
    n          : number of Poisson trains
    myseed     : random seed. int or boolean

  Returns:
    pre_spike_train : spike train matrix, ith row represents whether
                      there is a spike in ith spike train over time
                      (1 if spike, 0 otherwise)
  """

  # Retrieve simulation parameters
  dt, range_t = pars['dt'], pars['range_t']
  Lt = range_t.size

  # set random seed
  if myseed:
      np.random.seed(seed=myseed)
  else:
      np.random.seed()

  # generate uniformly distributed random variables
  u_rand = np.random.rand(n, Lt)

  # generate Poisson train
  poisson_train = 1. * (u_rand < rate * (dt / 1000.))

  return poisson_train


def default_pars(**kwargs):
  pars = {}

  ### typical neuron parameters###
  pars['V_th'] = -55.    # spike threshold [mV]
  pars['V_reset'] = -75. # reset potential [mV]
  pars['tau_m'] = 10.    # membrane time constant [ms]
  pars['g_L'] = 10.      # leak conductance [nS]
  pars['V_init'] = -65.  # initial potential [mV]
  pars['E_L'] = -75.     # leak reversal potential [mV]
  pars['tref'] = 2.      # refractory time (ms)

  ### simulation parameters ###
  pars['T'] = 400.  # Total duration of simulation [ms]
  pars['dt'] = .1   # Simulation time step [ms]

  ### external parameters if any ###
  for k in kwargs:
    pars[k] = kwargs[k]

  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # Vector of discretized time points [ms]

  return pars

# %% [markdown]
# ## Static synapses

# %% [markdown]
# ### Simulate synaptic conductance dynamics
# 
# - Synaptic input _in vivo_ consists of a mixture of **excitatory** neurotransmitters, which depolarizes the cell and drives it towards spike threshold, 
# - and **inhibitory** neurotransmitters that hyperpolarize it, driving it away from spike threshold. 
# - These chemicals cause specific ion channels on the postsynaptic neuron to open, resulting in a change in that neuron's conductance and, therefore, the flow of current in or out of the cell.
# 
# This process can be modeled by assuming that the presynaptic neuron's spiking activity produces transient changes in the postsynaptic neuron's conductance ($g_{\rm syn}(t)$). Typically, the conductance transient is modeled as an exponential function.
# 
# Such conductance transients can be generated using a simple ordinary differential equation (ODE):
# 
# <br>
# 
# \begin{equation}
# \frac{dg_{\rm syn}(t)}{dt} = \bar{g}_{\rm syn} \sum_k \delta(t-t_k) -g_{\rm syn}(t)/\tau_{\rm syn}
# \end{equation}
# 
# <br>
# 
# where $\bar{g}_{\rm syn}$ (often referred to as synaptic weight) is the maximum conductance elicited by each incoming spike, and $\tau_{\rm syn}$ is the synaptic time constant. Note that the summation runs over all spikes received by the neuron at time $t_k$.
# 
# 
# - $\frac{dg_{\rm syn}(t)}{dt}$ represents the rate of change of the synaptic conductance ($g_{\rm syn}(t)$) with respect to time ($t$). This term essentially describes how the synaptic conductance evolves over time.
# 
# - $\bar{g}_{\rm syn}$ represents the maximum synaptic conductance. This parameter signifies the maximum strength of the synaptic connection, i.e., the maximum conductance change that can occur due to synaptic input.
# 
# - $\delta(t - t_k)$ represents the Dirac delta function, which is zero everywhere except at $t = t_k$, where it equals infinity, with the integral over any interval containing $t_k$ equal to 1. In this context, $t_k$ represents the times at which presynaptic spikes occur. The Dirac delta function essentially models the instantaneous change in conductance caused by presynaptic spikes.
# 
# - $\tau_{\rm syn}$ represents the synaptic time constant, as discussed earlier. It characterizes how quickly the synaptic conductance changes in response to presynaptic spikes. A smaller $\tau_{\rm syn}$ value indicates faster changes in conductance, while a larger value indicates slower changes.
# 
# The equation describes how the synaptic conductance ($g_{\rm syn}(t)$) changes over time due to presynaptic spikes. The first term on the right-hand side represents the instantaneous increase in conductance whenever a presynaptic spike occurs. The second term represents the decay of the synaptic conductance over time, with a rate determined by the synaptic time constant $\tau_{\rm syn}$. This decay term ensures that the conductance returns to baseline in the absence of further presynaptic activity.
# 
# 
# 
# 
# Ohm's law allows us to convert conductance changes to the current as:
# 
# <br>
# 
# \begin{equation}
# I_{\rm syn}(t) = g_{\rm syn}(t)(V(t)-E_{\rm syn})  \\
# \end{equation}
# 
# <br>
# 
# The reversal potential $E_{\rm syn}$ determines the direction of current flow and the excitatory or inhibitory nature of the synapse.
# 
# **Thus, incoming spikes are filtered by an exponential-shaped kernel, effectively low-pass filtering the input. In other words, synaptic input is not white noise, but it is, in fact, colored noise, where the color (spectrum) of the noise is determined by the synaptic time constants of both excitatory and inhibitory synapses.**
# 
# In a neuronal network, the total synaptic input current $I_{\rm syn}$ is the sum of both excitatory and inhibitory inputs. Assuming the total excitatory and inhibitory conductances received at time $t$ are $g_E(t)$ and $g_I(t)$, and their corresponding reversal potentials are $E_E$ and $E_I$, respectively, then the total synaptic current can be described as:
# 
# <br>
# 
# \begin{equation}
# I_{\rm syn}(V(t),t) = -g_E(t) (V-E_E) - g_I(t) (V-E_I)
# \end{equation}
# 
# <br>
# 
# Accordingly, the membrane potential dynamics of the LIF neuron under synaptic current drive become:
# 
# <br>
# 
# \begin{equation}
# \tau_m\frac{dV(t)}{dt} = -(V(t)-E_L) - \frac{g_E(t)}{g_L} (V(t)-E_E) - \frac{g_I(t)}{g_L} (V(t)-E_I) + \frac{I_{\rm inj}}{g_L}\quad (2)
# \end{equation}
# 
# <br>
# 
# $I_{\rm inj}$ is an external current injected in the neuron, which is under experimental control; it can be GWN, DC, or anything else.
# 
# We will use Eq. (2) to simulate the conductance-based LIF neuron model below.
# 
# 

# %% [markdown]
# ### Simulate LIF neuron with conductance-based synapses
# 
# This is an important function:

# %%
# @markdown Execute this cell to get a function for conductance-based LIF neuron (run_LIF_cond)

def run_LIF_cond(pars, I_inj, pre_spike_train_ex, pre_spike_train_in):
  """
  Conductance-based LIF dynamics

  Args:
    pars               : parameter dictionary
    I_inj              : injected current [pA]. The injected current here
                         can be a value or an array
    pre_spike_train_ex : spike train input from presynaptic excitatory neuron
    pre_spike_train_in : spike train input from presynaptic inhibitory neuron

  Returns:
    rec_spikes         : spike times
    rec_v              : mebrane potential
    gE                 : postsynaptic excitatory conductance
    gI                 : postsynaptic inhibitory conductance

  """

  # Retrieve parameters
  V_th, V_reset = pars['V_th'], pars['V_reset']
  tau_m, g_L = pars['tau_m'], pars['g_L']
  V_init, E_L = pars['V_init'], pars['E_L']
  gE_bar, gI_bar = pars['gE_bar'], pars['gI_bar']
  VE, VI = pars['VE'], pars['VI']
  tau_syn_E, tau_syn_I = pars['tau_syn_E'], pars['tau_syn_I']
  tref = pars['tref']
  dt, range_t = pars['dt'], pars['range_t']
  Lt = range_t.size

  # Initialize
  tr = 0.
  v = np.zeros(Lt)
  v[0] = V_init
  gE = np.zeros(Lt)
  gI = np.zeros(Lt)
  Iinj = I_inj * np.ones(Lt)  # ensure Iinj has length Lt

  if pre_spike_train_ex.max() == 0:
    pre_spike_train_ex_total = np.zeros(Lt)
  else:
    pre_spike_train_ex_total = pre_spike_train_ex.sum(axis=0) * np.ones(Lt)

  if pre_spike_train_in.max() == 0:
    pre_spike_train_in_total = np.zeros(Lt)
  else:
    pre_spike_train_in_total = pre_spike_train_in.sum(axis=0) * np.ones(Lt)

  # simulation
  rec_spikes = []  # recording spike times
  for it in range(Lt - 1):
    if tr > 0:
      v[it] = V_reset
      tr = tr - 1
    elif v[it] >= V_th:  # reset voltage and record spike event
      rec_spikes.append(it)
      v[it] = V_reset
      tr = tref / dt

    # update the synaptic conductance
    gE[it + 1] = gE[it] - (dt / tau_syn_E) * gE[it] + gE_bar * pre_spike_train_ex_total[it + 1]
    gI[it + 1] = gI[it] - (dt / tau_syn_I) * gI[it] + gI_bar * pre_spike_train_in_total[it + 1]

    # calculate the increment of the membrane potential
    dv = (dt / tau_m) * (-(v[it] - E_L) \
                         - (gE[it + 1] / g_L) * (v[it] - VE) \
                         - (gI[it + 1] / g_L) * (v[it] - VI) + Iinj[it] / g_L)

    # update membrane potential
    v[it+1] = v[it] + dv

  rec_spikes = np.array(rec_spikes) * dt

  return v, rec_spikes, gE, gI


print(help(run_LIF_cond))

# %% [markdown]
# **Free Membrane Potential (FMP)** -- the membrane potential of the neuron when its spike threshold is removed
# 
# - calculating this quantity allows us to get an idea of how strong the input is
# - interested in knowing the mean and standard deviation (std.) of the FMP.

# %%
# Get default parameters
pars = default_pars(T=1000.)

# Add parameters
pars['gE_bar'] = 2.4    # [nS]
pars['VE'] = 0.         # [mV] excitatory reversal potential
pars['tau_syn_E'] = 2.  # [ms]
pars['gI_bar'] = 2.4    # [nS]
pars['VI'] = -80.       # [mV] inhibitory reversal potential
pars['tau_syn_I'] = 5.  # [ms]

# Generate presynaptic spike trains
pre_spike_train_ex = Poisson_generator(pars, rate=10, n=80)
pre_spike_train_in = Poisson_generator(pars, rate=10, n=20)

# Simulate conductance-based LIF model
v, rec_spikes, gE, gI = run_LIF_cond(pars, 0, pre_spike_train_ex,
                                     pre_spike_train_in)

# Show spikes more clearly by setting voltage high
dt, range_t = pars['dt'], pars['range_t']
if rec_spikes.size:
  sp_num = (rec_spikes / dt).astype(int) - 1
  v[sp_num] = 10   # draw nicer spikes

####################################################################
## TODO for students: measure the free membrane potential
# In order to measure the free membrane potential, first,
# you should prevent the firing of the LIF neuron
# How to prevent a LIF neuron from firing? Increase the threshold pars['V_th'].
#raise NotImplementedError("Student exercise: change threshold and calculate FMP")
####################################################################

# Change the threshold
pars['V_th'] = 1000

# Calculate FMP
v_fmp, _, _, _ = run_LIF_cond(pars, 0, pre_spike_train_ex, pre_spike_train_in)

my_illus_LIFSYN(pars, v_fmp, v)

# %%
# @markdown Make sure you execute this cell to enable the widget!

my_layout.width = '450px'
@widgets.interact(
    inh_rate=widgets.FloatSlider(20., min=10., max=60., step=5.,
                                 layout=my_layout),
    exc_rate=widgets.FloatSlider(10., min=2., max=20., step=2.,
                                 layout=my_layout)
)


def EI_isi_regularity(exc_rate, inh_rate):

  pars = default_pars(T=1000.)
  # Add parameters
  pars['gE_bar'] = 3.     # [nS]
  pars['VE'] = 0.         # [mV] excitatory reversal potential
  pars['tau_syn_E'] = 2.  # [ms]
  pars['gI_bar'] = 3.     # [nS]
  pars['VI'] = -80.       # [mV] inhibitory reversal potential
  pars['tau_syn_I'] = 5.  # [ms]

  pre_spike_train_ex = Poisson_generator(pars, rate=exc_rate, n=80)
  pre_spike_train_in = Poisson_generator(pars, rate=inh_rate, n=20)  # 4:1

  # Lets first simulate a neuron with identical input but with no spike
  # threshold by setting the threshold to a very high value
  # so that we can look at the free membrane potential
  pars['V_th'] = 1e3
  v_fmp, rec_spikes, gE, gI = run_LIF_cond(pars, 0, pre_spike_train_ex,
                                           pre_spike_train_in)

  # Now simulate a LIP with a regular spike threshold
  pars['V_th'] = -55.
  v, rec_spikes, gE, gI = run_LIF_cond(pars, 0, pre_spike_train_ex,
                                       pre_spike_train_in)
  dt, range_t = pars['dt'], pars['range_t']
  if rec_spikes.size:
      sp_num = (rec_spikes / dt).astype(int) - 1
      v[sp_num] = 10  # draw nicer spikes

  spike_rate = 1e3 * len(rec_spikes) / pars['T']

  cv_isi = 0.
  if len(rec_spikes) > 3:
    isi = np.diff(rec_spikes)
    cv_isi = np.std(isi) / np.mean(isi)

  print('\n')
  plt.figure(figsize=(15, 10))
  plt.subplot(211)
  plt.text(500, -35, f'Spike rate = {spike_rate:.3f} (sp/s), Mean of Free Mem Pot = {np.mean(v_fmp):.3f}',
           fontsize=16, fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom')
  plt.text(500, -38.5, f'CV ISI = {cv_isi:.3f}, STD of Free Mem Pot = {np.std(v_fmp):.3f}',
           fontsize=16, fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom')

  plt.plot(pars['range_t'], v_fmp, 'r', lw=1.,
           label='Free mem. pot.', zorder=2)
  plt.plot(pars['range_t'], v, 'b', lw=1.,
           label='mem. pot with spk thr', zorder=1, alpha=0.7)
  plt.axhline(pars['V_th'], 0, 1, color='k', lw=1., ls='--',
              label='Spike Threshold', zorder=1)
  plt.axhline(np.mean(v_fmp),0, 1, color='r', lw=1., ls='--',
              label='Mean Free Mem. Pot.', zorder=1)
  plt.ylim(-76, -39)
  plt.xlabel('Time (ms)')
  plt.ylabel('V (mV)')
  plt.legend(loc=[1.02, 0.68])

  plt.subplot(223)
  plt.plot(pars['range_t'][::3], gE[::3], 'r', lw=1)
  plt.xlabel('Time (ms)')
  plt.ylabel(r'$g_E$ (nS)')

  plt.subplot(224)
  plt.plot(pars['range_t'][::3], gI[::3], 'b', lw=1)
  plt.xlabel('Time (ms)')
  plt.ylabel(r'$g_I$ (nS)')

  plt.tight_layout()
  plt.show()

# %% [markdown]
# 1. How much can you increase the spike pattern variability? Under what condition(s) might the neuron respond with Poisson-type spikes? Note that we injected Poisson-type spikes. (Think of the answer in terms of the ratio of the exc. and inh. input spike rates.)
# 
# 2. Link to the balance of excitation and inhibition: one of the definitions of excitation and inhibition balance is that mean free membrane potential remains constant as excitatory and inhibitory input rates are increased. What do you think happens to the neuron firing rate as we change excitatory and inhibitory rates while keeping the neuron in balance?

# %% [markdown]
# 1. We can push the neuron to spike almost like a Poisson neuron. Of course given
# that there is a refractoriness it will never spike completely like a Poisson process.
# Poisson type spike irregularity will be achieved when mean is small (far from the
# spike threshold) and fluctuations are large. This will achieved when excitatory
# and inhibitory rates are balanced -- i.e. ratio of exc and inh. spike rate is
# constant as you vary the input rate.
# 
# 2. Firing rate will increase because fluctuations will increase as we increase
# exc. and inh. rates. But if synapses are modeled as conductance as opposed to
# currents, fluctuations may start decrease at high input rates because neuron time
# constant will drop.

# %% [markdown]
# ## Short-term synaptic plasticity

# %% [markdown]
# **Short-term plasticity (STP):** a phenomenon in which synaptic efficacy changes over time in a way that reflects the history of presynaptic activity； **Short-Term Depression (STD)** and **Short-Term Facilitation (STF)**
# 
# - based on the concept of a limited pool of **synaptic resources available for transmission ($R$)**, such as, for example, the overall amount of synaptic vesicles at the presynaptic terminals.
# - the amount of presynaptic resource changes in a dynamic fashion depending on the recent history of spikes
# 
# Following a presynaptic spike, (i) the fraction **$u$ (release probability)** of the available pool to be utilized increases due to spike-induced calcium influx to the presynaptic terminal, after which (ii) $u$ is consumed to increase the post-synaptic conductance. Between spikes, $u$ decays back to zero with time constant $\tau_f$ and $R$ recovers to 1 with time constant $\tau_d$. In summary, the dynamics of excitatory (subscript $E$) STP are given by:
# 
# <br>
# 
# \begin{equation}
# \frac{du_E}{dt} = -\frac{u_E}{\tau_f} + U_0(1-u_E^-)\delta(t-t_{\rm sp}) 
# \end{equation}
# 
# \begin{equation}
# \frac{dR_E}{dt} = \frac{1-R_E}{\tau_d} - u_E^+ R_E^- \delta(t-t_{\rm sp}) 
# \end{equation}
# 
# \begin{equation}
# \frac{dg_E(t)}{dt} = -\frac{g_E}{\tau_E} + \bar{g}_E u_E^+ R_E^- \delta(t-t_{\rm sp})
# \end{equation}
# 
# where $U_0$ is a constant determining the increment of $u$ produced by a spike. $u_E^-$ and $R_E^-$ denote the corresponding values just before the spike arrives, whereas $u_E^+$ refers to the moment right after the spike. $\bar{g}_E$ denotes the maximum excitatory conductance, and $g_E(t)$ is calculated for all spiketimes $k$, and decays over time with a time constant $\tau_{E}$. Similarly, one can obtain the dynamics of inhibitory STP (i.e., by replacing the subscript $E$ with $I$).
# 
# The interplay between the dynamics of $u$ and $R$ determines whether the joint effect of $uR$ is dominated by *depression* or *facilitation*. 
# - In the parameter regime of $\tau_d \gg \tau_f$  and for large $U_0$, an initial spike incurs a large drop in $R$ that takes a long time to recover; therefore, the synapse is STD-dominated. 
# - In the regime of $\tau_d \ll \tau_f$ and for small $U_0$, the synaptic efficacy is increased gradually by spikes, and consequently, the synapse is STF-dominated. This phenomenological model successfully reproduces the kinetic dynamics of depressed and facilitated synapses observed in many cortical areas.

# %%
# @markdown Execute this cell to get helper function `dynamic_syn`

def dynamic_syn(g_bar, tau_syn, U0, tau_d, tau_f, pre_spike_train, dt):
  """
  Short-term synaptic plasticity

  Args:
    g_bar           : synaptic conductance strength
    tau_syn         : synaptic time constant [ms]
    U0              : synaptic release probability at rest
    tau_d           : synaptic depression time constant of x [ms]
    tau_f           : synaptic facilitation time constantr of u [ms]
    pre_spike_train : total spike train (number) input
                      from presynaptic neuron
    dt              : time step [ms]

  Returns:
    u               : usage of releasable neurotransmitter
    R               : fraction of synaptic neurotransmitter resources available
    g               : postsynaptic conductance

  """

  Lt = len(pre_spike_train)
  # Initialize
  u = np.zeros(Lt)
  R = np.zeros(Lt)
  R[0] = 1.
  g = np.zeros(Lt)

  # simulation
  for it in range(Lt - 1):
    # Compute du
    du = -(dt / tau_f) * u[it] + U0 * (1.0 - u[it]) * pre_spike_train[it + 1]
    u[it + 1] = u[it] + du
    # Compute dR
    dR = (dt / tau_d) * (1.0 - R[it]) - u[it + 1] * R[it] * pre_spike_train[it + 1]
    R[it + 1] = R[it] + dR
    # Compute dg
    dg = -(dt / tau_syn) * g[it] + g_bar * R[it] * u[it + 1] * pre_spike_train[it + 1]
    g[it + 1] = g[it] + dg

  return u, R, g

help(dynamic_syn)

# %% [markdown]
# ### Short-term synaptic depression (STD)

# %%
# @markdown STD conductance ratio with different input rate
# Regular firing rate
input_rate = np.arange(5., 40.1, 5.)
g_1 = np.zeros(len(input_rate))  # record the the PSP at 1st spike
g_2 = np.zeros(len(input_rate))  # record the the PSP at 10th spike

for ii in range(len(input_rate)):
   g_1[ii], g_2[ii] = my_illus_STD(Poisson=False, rate=input_rate[ii],
                                   plot_out=False, U0=0.5, tau_d=100., tau_f=50)

plt.figure(figsize=(11, 4.5))

plt.subplot(121)
plt.plot(input_rate, g_1, 'm-o', label='1st Spike')
plt.plot(input_rate, g_2, 'c-o', label='10th Spike')
plt.xlabel('Rate [Hz]')
plt.ylabel('Conductance [nS]')
plt.legend()

plt.subplot(122)
plt.plot(input_rate, g_2 / g_1, 'b-o')
plt.xlabel('Rate [Hz]')
plt.ylabel(r'Conductance ratio $g_{10}/g_{1}$')\

plt.tight_layout()
plt.show()

# %% [markdown]
# As we increase the input rate the ratio of the first to tenth spike is increased, because the tenth spike conductance becomes smaller. This is a clear evidence of synaptic depression, as using the same amount of current has a smaller effect on the neuron.

# %% [markdown]
# ### Short-term synaptic facilitation (STF)

# %%
# @markdown STF conductance ratio with different input rates
# Regular firing rate
input_rate = np.arange(2., 40.1, 2.)
g_1 = np.zeros(len(input_rate))  # record the the PSP at 1st spike
g_2 = np.zeros(len(input_rate))  # record the the PSP at 10th spike

for ii in range(len(input_rate)):
   g_1[ii], g_2[ii] = my_illus_STD(Poisson=False, rate=input_rate[ii],
                                   plot_out=False,
                                   U0=0.2, tau_d=100., tau_f=750.)

plt.figure(figsize=(11, 4.5))

plt.subplot(121)
plt.plot(input_rate, g_1, 'm-o', label='1st Spike')
plt.plot(input_rate, g_2, 'c-o', label='10th Spike')
plt.xlabel('Rate [Hz]')
plt.ylabel('Conductance [nS]')
plt.legend()

plt.subplot(122)
plt.plot(input_rate, g_2 / g_1, 'b-o',)
plt.xlabel('Rate [Hz]')
plt.ylabel(r'Conductance ratio $g_{10}/g_{1}$')

plt.tight_layout()
plt.show()

# %% [markdown]
# Why does the ratio of the first and tenth spike conductance changes in a non-monotonic fashion for synapses with STF, even though it decreases monotonically for synapses with STD?
# 
# Because we have a facilitatory synapses, as the input rate increases synaptic
# resources released per spike also increase. Therefore, we expect that the synaptic
# conductance will increase with input rate. However, total synaptic resources are
# finite. And they recover in a finite time. Therefore, at high frequency inputs
# synaptic resources are rapidly deleted at a higher rate than their recovery, so
# after first few spikes, only a small number of synaptic resources are left. This
# results in decrease in the steady-state synaptic conductance at high frequency inputs.
# 

# %%



