# %% packages
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

# %% data import
cars_file = "https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv"
cars = pd.read_csv(cars_file)
cars.head()

# %% visualise the model
sns.scatterplot(x="wt", y="mpg", data=cars)
sns.regplot(x="wt", y="mpg", data=cars)

# %% convert data to tensor
X_list = cars.wt.values  # This is the first data
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)  # Convert it to numpy array
print(X_np)
print(X_np.shape)
# Target variable
y_list = cars.mpg.values.tolist()
X = torch.from_numpy(X_np)  # Can use this function if the data is already in numpy format
y = torch.tensor(y_list)  # Can use this function if the data is already in list format
# %% training
# Weight
w = torch.randn(1, requires_grad=True, dtype=torch.float32)
# Bias
b = torch.randn(1, requires_grad=True, dtype=torch.float32)

num_epochs = 1000
learning_rate = 0.001
# Data is fed to the neural network for training. Once all the data is provided to the neural network, it is called one epoch.
# Then the weights are adapted and the next iteration starts. This is the next epoch.

for epoch in range(num_epochs):
    for i in range(len(X)):  # This means we have a batch size of 1
        # Forward pass, we will calculate the predicted value
        # X at the i-th position is multiplied by the weight tensor and added to the bias tnesor
        y_pred = X[i] * w + b

        # Loss calculation
        loss_tensor = torch.pow(y_pred - y[i], 2)

        # Backward pass
        loss_tensor.backward()

        # extract the losses
        loss_value = loss_tensor.data[0]

        # Update the weights and biases
        with torch.no_grad():
            # Here we deactivate the auto gradient, because we want to use the calculated gradients and assign them to the tensors
            w -= w.grad * learning_rate
            b -= b.grad * learning_rate
            # We set the gradients to zero, because we want to calculate the new gradients in the next iteration
            # Underscores are used at the end of the function names to indicate that the function is performed in place
            w.grad.zero_()
            b.grad.zero_()
        print(loss_value)
# %% check results
# %%

# %% (Statistical) Linear Regression


# %% create graph visualisation
# make sure GraphViz is installed (https://graphviz.org/download/)
# if not computer restarted, append directly to PATH variable
# import os
# from torchviz import make_dot
# os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'
# make_dot(loss_tensor)
# %%
