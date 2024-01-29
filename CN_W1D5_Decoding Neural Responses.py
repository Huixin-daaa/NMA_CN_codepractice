# %% [markdown]
# ## Intro

# %% [markdown]
# - A model that **decodes** a variable from neural activity can tell us **how much information** a brain area contains about that variable. 
# - An **encoding model** is a model from an input variable, like visual stimulus, to neural activity. The encoding model is meant to approximate the same transformation that the brain performs on input variables and therefore help us understand **how the brain represents information**.

# %% [markdown]
# ## Setup

# %%
# Imports
import os
import numpy as np

import torch
from torch import nn
from torch import optim

import matplotlib as mpl
from matplotlib import pyplot as plt

# %%
# @title Figure Settings
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

%config InlineBackend.figure_format = 'retina'
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

# %%
# @title Plotting Functions

def plot_data_matrix(X, ax, show=False):
  """Visualize data matrix of neural responses using a heatmap

  Args:
    X (torch.Tensor or np.ndarray): matrix of neural responses to visualize
        with a heatmap
    ax (matplotlib axes): where to plot
    show (boolean): enable plt.show()

  """

  cax = ax.imshow(X, cmap=mpl.cm.pink, vmin=np.percentile(X, 1),
                  vmax=np.percentile(X, 99))
  cbar = plt.colorbar(cax, ax=ax, label='normalized neural response')

  ax.set_aspect('auto')
  ax.set_xticks([])
  ax.set_yticks([])
  if show:
    plt.show()


def plot_train_loss(train_loss):
  plt.plot(train_loss)
  plt.xlim([0, None])
  plt.ylim([0, None])
  plt.xlabel('iterations of gradient descent')
  plt.ylabel('mean squared error')
  plt.show()

# %%
# @title Helper Functions

def load_data(data_name, bin_width=1):
  """Load mouse V1 data from Stringer et al. (2019)

  Data from study reported in this preprint:
  https://www.biorxiv.org/content/10.1101/679324v2.abstract

  These data comprise time-averaged responses of ~20,000 neurons
  to ~4,000 stimulus gratings of different orientations, recorded
  through Calcium imaging. The responses have been normalized by
  spontaneous levels of activity and then z-scored over stimuli, so
  expect negative numbers. They have also been binned and averaged
  to each degree of orientation.

  This function returns the relevant data (neural responses and
  stimulus orientations) in a torch.Tensor of data type torch.float32
  in order to match the default data type for nn.Parameters in
  Google Colab.

  This function will actually average responses to stimuli with orientations
  falling within bins specified by the bin_width argument. This helps
  produce individual neural "responses" with smoother and more
  interpretable tuning curves.

  Args:
    bin_width (float): size of stimulus bins over which to average neural
      responses

  Returns:
    resp (torch.Tensor): n_stimuli x n_neurons matrix of neural responses,
        each row contains the responses of each neuron to a given stimulus.
        As mentioned above, neural "response" is actually an average over
        responses to stimuli with similar angles falling within specified bins.
    stimuli: (torch.Tensor): n_stimuli x 1 column vector with orientation
        of each stimulus, in degrees. This is actually the mean orientation
        of all stimuli in each bin.

  """
  with np.load(data_name) as dobj:
    data = dict(**dobj)
  resp = data['resp']
  stimuli = data['stimuli']

  if bin_width > 1:
    # Bin neural responses and stimuli
    bins = np.digitize(stimuli, np.arange(0, 360 + bin_width, bin_width))
    stimuli_binned = np.array([stimuli[bins == i].mean() for i in np.unique(bins)])
    resp_binned = np.array([resp[bins == i, :].mean(0) for i in np.unique(bins)])
  else:
    resp_binned = resp
    stimuli_binned = stimuli

  # Return as torch.Tensor
  resp_tensor = torch.tensor(resp_binned, dtype=torch.float32)
  stimuli_tensor = torch.tensor(stimuli_binned, dtype=torch.float32).unsqueeze(1)  # add singleton dimension to make a column vector

  return resp_tensor, stimuli_tensor


def get_data(n_stim, train_data, train_labels):
  """ Return n_stim randomly drawn stimuli/resp pairs

  Args:
    n_stim (scalar): number of stimuli to draw
    resp (torch.Tensor):
    train_data (torch.Tensor): n_train x n_neurons tensor with neural
      responses to train on
    train_labels (torch.Tensor): n_train x 1 tensor with orientations of the
      stimuli corresponding to each row of train_data, in radians

  Returns:
    (torch.Tensor, torch.Tensor): n_stim x n_neurons tensor of neural responses and n_stim x 1 of orientations respectively
  """
  n_stimuli = train_labels.shape[0]
  istim = np.random.choice(n_stimuli, n_stim)
  r = train_data[istim]  # neural responses to this stimulus
  ori = train_labels[istim]  # true stimulus orientation

  return r, ori

# %%
# @title Data retrieval and loading
import hashlib
import requests

fname = "W3D4_stringer_oribinned1.npz"
url = "https://osf.io/683xc/download"
expected_md5 = "436599dfd8ebe6019f066c38aed20580"

if not os.path.isfile(fname):
  try:
    r = requests.get(url)
  except requests.ConnectionError:
    print("!!! Failed to download data !!!")
  else:
    if r.status_code != requests.codes.ok:
      print("!!! Failed to download data !!!")
    elif hashlib.md5(r.content).hexdigest() != expected_md5:
      print("!!! Data download appears corrupted !!!")
    else:
      with open(fname, "wb") as fid:
        fid.write(r.content)

# %% [markdown]
# ## Load, split and visualize data

# %%
# @markdown Execute this cell to load and visualize data

# Load data
resp_all, stimuli_all = load_data(fname)  # argument to this function specifies bin width
n_stimuli, n_neurons = resp_all.shape

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 6, 5))

# Visualize data matrix
plot_data_matrix(resp_all[:, :100].T, ax1)  # plot responses of first 100 neurons
ax1.set_xlabel('stimulus')
ax1.set_ylabel('neuron')

# Plot tuning curves of three random neurons
ineurons = np.random.choice(n_neurons, 3, replace=False)  # pick three random neurons
ax2.plot(stimuli_all, resp_all[:, ineurons])
ax2.set_xlabel('stimulus orientation ($^o$)')
ax2.set_ylabel('neural response')
ax2.set_xticks(np.linspace(0, 360, 5))

fig.suptitle(f'{n_neurons} neurons in response to {n_stimuli} stimuli')
fig.tight_layout()
plt.show()

# %%
# @markdown Execute this cell to split into training and test sets

# Set random seeds for reproducibility
np.random.seed(4)
torch.manual_seed(4)

# Split data into training set and testing set
n_train = int(0.6 * n_stimuli)  # use 60% of all data for training set
ishuffle = torch.randperm(n_stimuli)
itrain = ishuffle[:n_train]  # indices of data samples to include in training set
itest = ishuffle[n_train:]  # indices of data samples to include in testing set
stimuli_test = stimuli_all[itest]
resp_test = resp_all[itest]
stimuli_train = stimuli_all[itrain]
resp_train = resp_all[itrain]

# %% [markdown]
# ## Deep feed-forward networks in pytorch

# %% [markdown]
# ### Introduction to PyTorch

# %%
class DeepNet(nn.Module):
  """Deep Network with one hidden layer

  Args:
    n_inputs (int): number of input units
    n_hidden (int): number of units in hidden layer

  Attributes:
    in_layer (nn.Linear): weights and biases of input layer
    out_layer (nn.Linear): weights and biases of output layer

  """

  def __init__(self, n_inputs, n_hidden):
    super().__init__()  # needed to invoke the properties of the parent class nn.Module
    self.in_layer = nn.Linear(n_inputs, n_hidden) # neural activity --> hidden units
    self.out_layer = nn.Linear(n_hidden, 1) # hidden units --> output

  def forward(self, r):
    """Decode stimulus orientation from neural responses

    Args:
      r (torch.Tensor): vector of neural responses to decode, must be of
        length n_inputs. Can also be a tensor of shape n_stimuli x n_inputs,
        containing n_stimuli vectors of neural responses

    Returns:
      torch.Tensor: network outputs for each input provided in r. If
        r is a vector, then y is a 1D tensor of length 1. If r is a 2D
        tensor then y is a 2D tensor of shape n_stimuli x 1.

    """
    h = self.in_layer(r)  # hidden representation
    y = self.out_layer(h)
    return y

# %% [markdown]
# ### Activation functions

# %%
class DeepNetReLU(nn.Module):
  """ network with a single hidden layer h with a RELU """

  def __init__(self, n_inputs, n_hidden):
    super().__init__()  # needed to invoke the properties of the parent class nn.Module
    self.in_layer = nn.Linear(n_inputs, n_hidden) # neural activity --> hidden units
    self.out_layer = nn.Linear(n_hidden, 1) # hidden units --> output

  def forward(self, r):

    ############################################################################
    ## TO DO for students: write code for computing network output using a
    ## rectified linear activation function for the hidden units
    # Fill out function and remove
    #raise NotImplementedError("Student exercise: complete DeepNetReLU forward")
    ############################################################################

    # format: 
    # output = layer(input)
    h = self.in_layer(r) # h is size (n_inputs, n_hidden)
    y = self.out_layer(torch.relu(h)) # y is size (n_inputs, 1)


    return y


# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# Initialize a deep network with M=200 hidden units
net = DeepNetReLU(n_neurons, 200)

# Get neural responses (r) to and orientation (ori) to one stimulus in dataset
r, ori = get_data(1, resp_train, stimuli_train)  # using helper function get_data

# Decode orientation from these neural responses using initialized network
out = net(r)  # compute output from network, equivalent to net.forward(r)

print(f'decoded orientation: {out.item():.2f} degrees')
print(f'true orientation: {ori.item():.2f} degrees')

# %% [markdown]
# ## Loss functions and gradient descent

# %% [markdown]
# ### Loss functions

# %%
# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# Initialize a deep network with M=10 hidden units
net = DeepNetReLU(n_neurons, 10)

# Get neural responses to first 20 stimuli in the data set
r, ori = get_data(20, resp_train, stimuli_train)

# Decode orientation from these neural responses
out = net(r)

# Initialize PyTorch mean squared error loss function (Hint: look at nn.MSELoss)
loss_fn = nn.MSELoss()

# Evaluate mean squared error
# This loss function takes two inputs, the network output out and the true stimulus orientations ori and finds the mean squared error: loss = loss_fn(out, ori). 
# Specifically, it will take as arguments a batch of network outputs y1,y2,…,yP and corresponding target outputs y~1,y~2,…,y~P, and compute the mean squared error (MSE)
loss = loss_fn(out, ori)

print(f'mean squared error: {loss:.2f}')

# %% [markdown]
# ### Optimization with gradient descent

# %% [markdown]
# In gradient descent we compute the gradient of the loss function with respect to each parameter (all W’s and b’s). We then update the parameters by subtracting the learning rate times the gradient.

# %% [markdown]
# We'll use the **gradient descent (GD)** algorithm to modify our weights to reduce the loss function, which consists of iterating three steps. 
# 
# 1. **Evaluate the loss** on the training data,
# ```python
# out = net(train_data)
# loss = loss_fn(out, train_labels)
# ```
# where `train_data` are the network inputs in the training data (in our case, neural responses), and `train_labels` are the target outputs for each input (in our case, true stimulus orientations).
# 
# 2. **Compute the gradient of the loss** with respect to each of the network weights. In PyTorch, we can do this with the `.backward()` method of the loss `loss`. Note that the gradients of each parameter need to be cleared before calling `.backward()`, or else PyTorch will try to accumulate gradients across iterations. This can again be done using built-in optimizers via the method `.zero_grad()`. Putting these together we have
# ```python
# optimizer.zero_grad()
# loss.backward()
# ```
# 3. **Update the network weights** by descending the gradient. In Pytorch, we can do this using built-in optimizers. We'll use the `optim.SGD` optimizer which updates parameters along the negative gradient, scaled by a learning rate. To initialize this optimizer, we have to tell it
#   * which parameters to update, and
#   * what learning rate to use
# 
#   For example, to optimize *all* the parameters of a network `net` using a learning rate of .001, the optimizer would be initialized as follows
#   ```python
#   optimizer = optim.SGD(net.parameters(), lr=.001)
#   ```
#   where `.parameters()` is a method of the `nn.Module` class that returns a Python generator object over all the parameters of that `nn.Module` class (in our case, $\mathbf{W}^{in}, \mathbf{b}^{in}, \mathbf{W}^{out}, \mathbf{b}^{out}$).
# 
#   After computing all the parameter gradients in step 2, we can then update each of these parameters using the `.step()` method of this optimizer,
#   ```python
#   optimizer.step()
#   ```

# %% [markdown]
# ### Gradient descent in PyTorch

# %%
def train(net, loss_fn, train_data, train_labels,
          n_epochs=50, learning_rate=1e-4):
  """Run gradient descent to optimize parameters of a given network

  Args:
    net (nn.Module): PyTorch network whose parameters to optimize
    loss_fn: built-in PyTorch loss function to minimize
    train_data (torch.Tensor): n_train x n_neurons tensor with neural
      responses to train on
    train_labels (torch.Tensor): n_train x 1 tensor with orientations of the
      stimuli corresponding to each row of train_data
    n_epochs (int, optional): number of epochs of gradient descent to run
    learning_rate (float, optional): learning rate to use for gradient descent

  Returns:
    (list): training loss over iterations

  """

  # Initialize PyTorch SGD optimizer
  optimizer = optim.SGD(net.parameters(), lr=learning_rate)

  # Placeholder to save the loss at each iteration
  train_loss = []

  # Loop over epochs
  for i in range(n_epochs):

    ######################################################################
    ## TO DO for students: fill in missing code for GD iteration
    #raise NotImplementedError("Student exercise: write code for GD iterations")
    ######################################################################

    # compute network output from inputs in train_data
    out = net(train_data)  # compute network output from inputs in train_data

    # evaluate loss function
    loss = loss_fn(out, train_labels)

    # Clear previous gradients
    optimizer.zero_grad()

    # Compute gradients
    loss.backward()

    # Update weights
    optimizer.step()

    # Store current value of loss
    train_loss.append(loss.item())  # .item() needed to transform the tensor output of loss_fn to a scalar

    # Track progress
    if (i + 1) % (n_epochs // 5) == 0:
      print(f'iteration {i + 1}/{n_epochs} | loss: {loss.item():.3f}')

  return train_loss


# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# Initialize network with 10 hidden units
net = DeepNetReLU(n_neurons, 10)

# Initialize built-in PyTorch MSE loss function
loss_fn = nn.MSELoss()

# Run gradient descent on data
train_loss = train(net, loss_fn, resp_train, stimuli_train)

# Plot the training loss over iterations of GD
plot_train_loss(train_loss)

# %% [markdown]
# ### Backpropagation

# %% [markdown]
# The way that the gradients are calculated is called **backpropagation**. We have a loss function:
# 
# \begin{align}
# L &= (y - \tilde{y})^2 \\
# &= (\mathbf{W}^{out} \mathbf{h} - \tilde{y})^2
# \end{align}
# 
# where $\mathbf{h} = \phi(\mathbf{W}^{in} \mathbf{r} + \mathbf{b}^{in})$, and $\phi(\cdot)$ is the activation function, e.g., RELU.
# You may see that $\frac{\partial L}{\partial \mathbf{W}^{out}}$ is simple to calculate as it is on the outside of the equation (it is also a vector in this case, not a matrix, so the derivative is standard):
# 
# \begin{equation}
# \frac{\partial L}{\partial \mathbf{W}^{out}} = 2 (\mathbf{W}^{out} \mathbf{h} - \tilde{y})\mathbf{h}^\top
# \end{equation}
# 
# Now let's compute the derivative with respect to $\mathbf{W}^{in}$ using the chain rule. Note it is only positive if the output is positive due to the RELU activation function $\phi$. For the chain rule we need the derivative of the loss with respect to $\mathbf{h}$:
# 
# \begin{equation}
# \frac{\partial L}{\partial \mathbf{h}} = 2 \mathbf{W}^{out \top} (\mathbf{W}^{out} \mathbf{h} - \tilde{y})
# \end{equation}
# 
# Thus,
# 
# \begin{align}
# \frac{\partial L}{\partial \mathbf{W}^{in}} &= \begin{cases}
# \frac{\partial L}{\partial \mathbf{h}} \frac{\partial \mathbf{h}}{\partial \mathbf{W}^{in}}  & \text{if }  \mathbf{h} > 0 \\
# 0 & \text{otherwise}
# \end{cases} \\
# &= \begin{cases}
# 2 \mathbf{W}^{out \top} (\mathbf{W}^{out} \mathbf{h} - \tilde{y}) \mathbf{r}^\top  & \text{if }  \mathbf{h} > 0 \\
# 0 & \text{otherwise}
# \end{cases}
# \end{align}
# 
# Notice that:
# 
# \begin{equation}
# \frac{\partial \mathbf{h}}{\partial \mathbf{W}^{in}}=\mathbf{r}^\top \odot \phi^\prime
# \end{equation}
# 
# where $\odot$ denotes the Hadamard product (i.e., elementwise multiplication) and $\phi^\prime$ is the derivative of the activation function. In case of RELU:
# 
# \begin{align}
# \phi^\prime &= \begin{cases}
# 1  & \text{if }  \mathbf{h} > 0 \\
# 0 & \text{otherwise}
# \end{cases}
# \end{align}

# %% [markdown]
# ### Stochastic gradient descent (SGD) vs. gradient descent (GD)

# %% [markdown]
# - The key difference is in the very first step of each iteration, where in the GD algorithm we evaluate the loss *at every data sample in the training set*. 
# - In SGD, on the other hand, we evaluate the loss only at a random subset of data samples from the full training set, called a **mini-batch**. At each iteration, we randomly sample a mini-batch to perform steps 1-3 on. All the above equations still hold, but now the $P$ data samples $\mathbf{r}^{(n)}, \tilde{y}^{(n)}$ denote a mini-batch of $P$ random samples from the training set, rather than the whole training set.


