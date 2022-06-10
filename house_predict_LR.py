import numpy as np
import matplotlib.pyplot as plt

'''load the dataset separeted by delimeters'''
data = np.loadtxt("data.txt", delimiter = ",")

'''
data X for the square root and Y for the price
.reshape(X.size,1) for reshapig the Y without changing its value
.vstack vertically sequence the array 
.ones gives the array filled with ones and .T for transpose
'''
X = data[:, 0]
Y = data[:, 1].reshape(X.size, 1)
X = np.vstack((np.ones((X.size, )), X)).T

'''
.shape to see the shape of X and Y
'''
print(X.shape)
print(Y.shape)

plt.scatter(X[:, 1], Y)
plt.show()

'''
theta contains zeros of shape (2,1)
cost list is just to store the cost value at every iteration
y_pred is the dot product of X and theta
update : theta = theta - learning_rate*d_theta
returns theta and cost_list for plot
cost function determines the error between actual and predicted value
GD for eroor to be minimum

'''
def model(X, Y, learning_rate, iteration):
 m = Y.size
 theta = np.zeros((2, 1))
 cost_list = []

 for i in range(iteration):

    y_pred = np.dot(X, theta)
    cost = (1/(2*m))*np.sum(np.square(y_pred - Y))

    d_theta = (1/m)*np.dot(X.T, y_pred - Y)
    theta = theta - learning_rate*d_theta

    cost_list.append(cost)

 return theta, cost_list

'''
for iteration 100 and learning_rate to minimum
'''
iteration = 100
learning_rate = 0.00000005
theta, cost_list = model(X, Y, learning_rate = learning_rate,
iteration = iteration)


'''
Now predict the price in $ 
'''
new_houses = np.array([[1, 1547], [1, 1896], [1, 1934], [1,2800], [1, 3400], [1, 5000]])

for house in new_houses :
 print("Our model predicts the price of house with",house[1], "sq. ft. area as : $", round(np.dot(house, theta)[0],2))


'''
plot the curve showing cost vs iteration
'''
rng = np.arange(0, iteration)
plt.plot(cost_list, rng)
plt.show()