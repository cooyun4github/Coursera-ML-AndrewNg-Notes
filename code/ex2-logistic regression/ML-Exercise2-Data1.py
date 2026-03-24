import numpy
import pandas
import matplotlib.pyplot as plt

# 读样本数据
path = "code/ex2-logistic regression/ex2data1.txt"
data1 = pandas.read_csv(path, names=['Exam1', 'Exam2', 'Admitted'])
# print(data1.head())

# 看样本轮廓
positive = data1[data1['Admitted'] == 1]
negative = data1[data1['Admitted'] == 0]
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
# plt.show()

data1.insert(0, 'Ones', 1)
columns = data1.shape[1]
X = data1.iloc[:, 0:columns-1]
y = data1.iloc[:, columns-1:columns]
theta = numpy.zeros((1, columns-1))
X = numpy.matrix(X.values)
y = numpy.matrix(y.values)
print(X.shape, theta.shape, y.shape)

def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))

def error(X, y, theta):
    return sigmoid(X * theta.T) - y

# 这里的theta是fmin_tnc函数传入的参数，是把外部theta压扁后的一个一维数组，所以需要转成矩阵
# 初始theta：(3,1)，压扁后：(3,)，矩阵化：(1,3)，转置后：(3,1)
def cost(theta, X, y):
    theta = numpy.matrix(theta)
    first = numpy.multiply(-y, numpy.log(sigmoid(X * theta.T)))
    second = numpy.multiply((1 - y), numpy.log(1 - sigmoid(X * theta.T)))
    return numpy.sum(first - second) / (len(X))

def gradient(theta, X, y):
    theta = numpy.matrix(theta)
    return X.T * error(X, y, theta) / len(X)

import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print("Result:", result)
print(cost(result[0], X, y))

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = numpy.matrix(result[0])
print(theta_min.shape)
predictions = predict(theta_min, X)
correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct) * 100
print('accuracy = {0}%'.format(accuracy))

# x = numpy.linspace(data1.Exam1.min(), data1.Exam1.max(), 100)
# f = hTheta(x, result[0])

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data1.Exam1, data1.Exam2, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Exam1 Score')
# ax.set_ylabel('Exam2 Score')
# ax.set_title('Predicted Exam2 Score vs. Exam1 Score')
# plt.show()