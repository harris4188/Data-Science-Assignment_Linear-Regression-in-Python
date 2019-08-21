import numpy as np
import matplotlib.pyplot as plt

alpha = 0.9


def cost(X, T, y):
    return float((1/(2*y.size))*np.transpose(X*T-y)*(X*T-y))


def gradient_descent(X, T, y):
    count = 0
    m = y.size
    costs = []
    while count < 50:
        new_cost = cost(X, T, y)
        costs.append(new_cost)
        T = T - (alpha/m)*np.transpose(np.transpose(X*T - y)*X)
        print(T)
        print('New Cost:', new_cost)
        count += 1
    return costs, T

data = open('normalized_features.txt').read().splitlines()
mean_size = float(data[0])
std_size = float(data[1])
data[2] = data[2].split(' ')
for x in range(len(data[2])):
    data[2][x] = float(data[2][x])
mean_numberOfBedrooms = float(data[3])
std_numberOfBedrooms = float(data[4])
data[5] = data[5].split(' ')
for x in range(len(data[5])):
    data[5][x] = float(data[5][x])

price = []

data2 = open('ex1data2.txt').read().splitlines()
for x in range(len(data2)):
    data2[x] = data2[x].split(',')
    price.append(int(data2[x][2]))

size = np.asarray(data[2])
numberOfBedrooms = np.asarray(data[5])
price = np.asarray(price)
x0 = np.ones(len(size))
X = np.vstack((x0, size, numberOfBedrooms))
X = np.matrix(X)
X = np.transpose(X)

T = np.zeros(3)
T = np.matrix(T)
T = np.transpose(T)

print(X)
print(T)
print('\\\\\\\\\\\\\\\\\\')
print(X*T)

y = np.matrix(price)
y = np.transpose(y)
print(y)
print(cost(X, T, y))

J, new_T = gradient_descent(X, T, y)
numberOfIterations = np.arange(50)

print(J)

plt.xlabel('Number of Iterations')
plt.ylabel('Cost J')
plt.plot(numberOfIterations, J, 'green')
plt.show()

print(new_T)

test_x0 = 1
test_normalized_size = (1650 - mean_size)/std_size
test_normalized_numberOfBedrooms = (3 - mean_numberOfBedrooms)/std_numberOfBedrooms

test_X = np.vstack((test_x0, test_normalized_size, test_normalized_numberOfBedrooms))
test_X = np.matrix(test_X)
test_X = np.transpose(test_X)

normal_test_X = np.vstack((1, 1650, 3))
normal_test_X = np.matrix(normal_test_X)
normal_test_X = np.transpose(normal_test_X)

data3 = open('ex1data2.txt').read().splitlines()
normal_size = []
normal_numberOfBedrooms = []

for x in range(len(data3)):
    data3[x] = data3[x].split(',')
    normal_size.append(int(data3[x][0]))
    normal_numberOfBedrooms.append(int(data3[x][1]))

normal_x0 = x0
normal_size = np.asarray(normal_size)
normal_numberOfBedrooms = np.asarray(normal_numberOfBedrooms)

normal_X = np.vstack((normal_x0, normal_size, normal_numberOfBedrooms))
normal_X = np.matrix(normal_X)
normal_X = np.transpose(normal_X)


x_transpose = np.transpose(normal_X)
normal_new_T = np.linalg.pinv(x_transpose*normal_X)*x_transpose*y

print(normal_new_T)

print('Price using gradient descent: ', test_X*new_T)
print('Price using normal equation: ', normal_test_X*normal_new_T)
