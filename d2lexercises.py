from mxnet import np, npx, autograd
npx.set_np() ## When using tensors, we almost always invoke the set_np function. This is for compatibility.

x = np.arange(12)
print("tensor x is constructed as below")
print(x)
print("shape of tensor x is found as below")
print(x.shape)
print("size of tensor x is found as below")
print(x.size)

X = x.reshape(3, 4)
print("reshaped tensor x is shown as X and found as below")
print(X)
#print(x.reshape(-1, 4)) could be used for the same function
#print(x.reshape(3, -1)) could be used for the same function

print("Tensor of 1s with the shape of 2*3*4")
print(np.ones((2, 3, 4)))

print("Tensor with mean value= 0, standard deviation= 1, size= 3*4")
print(np.random.normal(0, 1, size=(3, 4)))

print("Tensor whose numerical values are set manually")
print(np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
print("Different operations on matrices elementwise")
print(x + y, x - y, x * y, x / y, x ** y)

print("Taking the exponential of every element in tensor x")
print(np.exp(x))

X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X,Y)
print("Concatenating (merging) X and Y vertically")
print(np.concatenate([X, Y], axis=0))
print("Concatenating (merging) X and Y horizontally")
print(np.concatenate([X, Y], axis=1))

print("Logical operations between X and Y elementwise")
print(X == Y)

print("Summing all the elements in the tensor yields a tensor with only one element")
print(X.sum())

#Broadcasting Mechanism
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
print("Tensor a")
print(a)
print("Tensor b")
print(b)
print("Summation of broadcasted entries a and b")
print(a+b)

print("Selecting the last row of tensor X")
print(X[-1])
print("Selecting the rows of tensor X from 1 to 3 (excluding 3)")
print(X[1:3])

print("Assigning a specific number to a specific location in the tensor X")
X[1, 2] = 9
print(X)

print("Assigning a specific number to a many locations in the tensor X")
X[0:2, :] = 12
print(X)

#Saving memory
print("In the following example we see how computer sets a new memory adress for a new operation")
before = id(Y)
Y = Y + X
print(id(Y) == before)
print(id(y))
print(before)

print("We will prove memory allocation as follows:")
Z = np.zeros_like(Y)
before = id(Z)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
print(id(Z)==before)

print("Another notation could be used as '+=' instead of working with another temporary tensor such as Z")
before = id(X)
X += Y
print(id(X) == before)

#Conversion to other Python Objects
print("Conversion to other Python Objects")
A = X.asnumpy()
B = np.array(A)
print(type(A), type(B))

a = np.array([3.5])
print(a, a.item(), float(a), int(a))

print("2.2 Data Preprocesing")

import os

def mkdir_if_not_exist(path): #@save

    """Make a directory if it does not exist."""
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)

print("Creating a .csv file and reading it")
data_file = '../data/house_tiny.csv'
mkdir_if_not_exist('../data')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # Column names
    f.write('NA,Pave,127500\n') # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd
data = pd.read_csv(data_file)
print(data)

print("Handling missing data 'NaN'")
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

print("Conversion to the Tensor Format")
X, y = np.array(inputs.values), np.array(outputs.values)
print(X,y)

print("Linear Algebra")
print("Scalars")
x = np.array(3.0)
y = np.array(2.0)
print(x + y, x * y, x / y, x ** y)

print("Vector")
x = np.arange(4)
print(x)
print("length of x:")
print(len(x))
print("shape of x:")
print(x.shape)

print("Assign a matrix and reshape it")
A = np.arange(20).reshape(5, 4)
print(A)

print("Assign a symmetric matrix and compare it with its transpozed version")
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B == B.T)#Comparing it with its transpoze

print("Tensors")
X = np.arange(24).reshape(2, 3, 4)
print(X)

print("Assign a copy of `A` to `B` by allocating new memory")
A = np.arange(20).reshape(5, 4)
B = A.copy() # Assign a copy of `A` to `B` by allocating new memory
print(A, A + B)

print("Calculation the sum of elements of a tensor")
x = np.arange(4)
print(x,x.sum())
print(A.shape, A.sum())

print("Summing each column")
A_sum_axis0 = A.sum(axis=0)
#if the axis=1 was used, it would sum up the elements of each row implicitly
print(A,A_sum_axis0, A_sum_axis0.shape)

print("it is the same thing to use .mean() and .sum/.size")
print(A.mean(), A.sum() / A.size)
print("Likewise, the function for calculating the mean can also reduce a tensor along the specified axes.")
print(A.mean(axis=0), A.sum(axis=0, keepdims=True) / A.shape[0])
#You don't have to use keepdims=True all the time. It just helps you to remain the dimensions as the original.

print("Non-reduction Sum")
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A / sum_A)
#we divided each element of a row by the mean of that particular row


print("Cumulative summation of each column elements",A.cumsum(axis=0))

print("Matrix-vector multiplication",np.dot(A, x))

print("Matrix-matrix multiplication")
B = np.ones(shape=(4, 3))
print(A.shape)
print(B.shape)
print(np.dot(A, B))

print("Norms")
u = np.array([3, -4])
print(np.linalg.norm(u)) #taking the square of each element, summing them all up and taking its square root.

print("Taking absolute value of each element and sum them:",np.abs(u).sum())

print("Taking the norm for 4*9 ones matrix",np.linalg.norm(np.ones((4, 9))))

print("Calculus")

print("Derivative of 3x^2-4x when x=1 by iteration")
from d2l import mxnet as d2l
from IPython import display
#here in this example, you can see how derivative works on each iteration. In each iteration, the exact result will come closer to the theorical result since h gets smaller.
def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h
h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

def use_svg_display(): #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)): #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
        ylim=None, xscale='linear', yscale='linear',
        fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()
    #Return True if `X` (tensor or list) has 1 axis

    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

import matplotlib.pyplot as plt
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
plt.show()

print("Automatic Differentiation")
x = np.arange(4.0)
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
print(x.grad)
with autograd.record():
    y = 2 * np.dot(x, x)
print(y)

# When we invoke `backward` on a vector-valued variable `y` (function of `x`),
# a new scalar variable is created by summing the elements in `y`. Then the
# gradient of that scalar variable with respect to `x` is computed
with autograd.record():
    y = x * x # `y` is a vector
y.backward()
print("Equals to y = sum(x * x) : ",x.grad) # Equals to y = sum(x * x)

print("Detaching Computation")
#Here, we can detach y to return a new variable u that has the same value as y but discards any
#information about how y was computed in the computational graph. In other words, the gradient
#will not flow backwards through u to x.
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
print(x.grad == u)

print("Computing the Gradient of Python Control Flow")
#The function below has just been copied and pasted from the d2l book. The key point here is that our ability to achieve
#to take gradient of a variable resulted by a control flow.
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()

print(a.grad == d / a)

from mxnet import random

print("Probability")
#note that fair_probs here is not a numpy array. If it would be a numpy array, the result of the multiplication with 6
# would be [1]
fair_probs = [1.0 / 6] * 6
print(np.random.multinomial(1, fair_probs))

print(np.random.multinomial(10, fair_probs))
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
print(counts / 1000)

counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
plt.show()