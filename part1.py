import numpy as np
import matplotlib.pyplot as plt

x0 = 1
t0 = 0
t1 = 0

alpha = 0.01


def h(x):
    return x0*t0 + x*t1


def cost(x, y):
    f = np.vectorize(h)
    o = f(x)
    return (1/(2*x.size))*np.sum((o-y)**2)


def gradient_descent(x, y):
    global t0, t1
    count = 0
    while count < 5000:
        old_cost = cost(x, y)
        f = np.vectorize(h)
        o = f(x)
        temp0 = t0 - alpha*(1/x.size)*np.sum(o - y)
        temp1 = t1 - alpha*(1/x.size)*np.sum((o - y)*x)
        t0 = temp0
        t1 = temp1
        new_cost = cost(x, y)
        print('t0:', t0)
        print('t1:', t1)
        print('Old Cost:', old_cost)
        print('New Cost:', new_cost)
        if new_cost >= old_cost:
            break
        count += 1
    return new_cost

data = open('ex1data1.txt').read().splitlines()
population = []
profit = []

for x in range(len(data)):
    data[x] = data[x].split(',')
    population.append(float(data[x][0]))
    profit.append(float(data[x][1]))

min_x = min(population)
max_x = max(population)
min_y = min(profit)
max_y = max(profit)

x = np.asarray(population)
y = np.asarray(profit)

min_cost = gradient_descent(x, y)

print('--------------------------------------')
print('Predicted Value of t0:', t0)
print('Predicted Value of t1:', t1)
print('Minimum Cost reached:', min_cost)

f = np.vectorize(h)
y2 = f(x)

plt.axis([min_x-2, max_x+2, min_y-2, max_y+2])
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(population, profit, 'rx', label='Training data')
plt.plot(x, y2, label='Linear regression')
plt.legend(loc='lower right')
plt.show()
