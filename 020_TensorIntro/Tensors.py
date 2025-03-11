# %% packages
import numpy as np
import seaborn as sns
from torch import Tensor, tensor

# %% create a tensor
x: Tensor = tensor(5.5)

# %% simple calculations
y = x + 10
print(y)

# %% automatic gradient calculation
print(x.requires_grad)  # check if requires_grad is true, false if not directly specified

x.requires_grad_()  # set requires grad to true, default True

# %% or set the flag directly during creation
x = tensor(2.0, requires_grad=True)
print(x.requires_grad)


# %% function for showing automatic gradient calculation
def y_function(val: float) -> float:
    return (val - 3) * (val - 6) * (val - 4)


x_range: np.ndarray = np.linspace(0, 10, 101)
print(x_range)
y_range = [y_function(i) for i in x_range]
sns.lineplot(x=x_range, y=y_range)

# %% define y as function of x
y: Tensor = (x - 3) * (x - 6) * (x - 4)
print(y)
# %%

# %% x -> y
# create a tensor with gradients enabled
x = tensor(1.0, requires_grad=True)
# create second tensor depending on first tensor
y = (x - 3) * (x - 6) * (x - 4)
# calculate gradients
y.backward()
# show gradient of first tensor
print(x.grad)
# %% x -> y -> z
x = tensor(1.0, requires_grad=True)
y = x**3
z = 5 * y - 4

# %%
z.backward()
print(x.grad)  # should be equal 5*3x**2
# %% more complex network
x11 = tensor(2.0, requires_grad=True)
x21 = tensor(3.0, requires_grad=True)
x12 = 5 * x11 - 3 * x21
x22 = 2 * x11**2 + 2 * x21
y = 4 * x12 + 3 * x22
y.backward()
print(x11.grad)
print(x21.grad)
# %%
