import numpy
import pandas
import matplotlib.pyplot as plt

ex1data2 = "code/ex1-linear regression/ex1data2.txt"
data2 = pandas.read_csv(ex1data2, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std()
data2.insert(0, 'Ones', 1)

columns = data2.shape[1]
X = data2.iloc[:, 0:columns-1]
y = data2.iloc[:, columns-1:columns]
theta = numpy.zeros((columns-1, 1))
X = numpy.matrix(X.values)
y = numpy.matrix(y.values)
print(X.shape, theta.shape, y.shape)
print(theta)

def hTheta(X, theta):
    return X * theta

def error(X, y, theta):
    return hTheta(X, theta) - y

def partialDerivative(X, y, theta):
    return X.T * error(X, y, theta) / len(X)

# cost函数只是为了观察cost变化曲线，并不参与梯度下降整个计算过程
def computeCost(X, y, theta):
    inner = numpy.power(((X * theta) - y), 2)
    return numpy.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    costs = numpy.zeros(iters)
    for i in range(iters):
        theta = theta - alpha * partialDerivative(X, y, theta)
        costs[i] = computeCost(X, y, theta)
    return theta, costs

iters = 1000
t, costs = gradientDescent(X, y, theta, 0.01, iters)
cost = computeCost(X, y, t)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(numpy.arange(iters), costs, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()