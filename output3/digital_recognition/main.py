#!/usr/bin/env python3

import csv
from numpy import zeros, random, multiply, add, exp, divide, append, shape, resize, reshape, insert, transpose, log, \
    sum, argmax, count_nonzero, copy, nan_to_num, array, loadtxt, float128, asarray
from scipy import optimize

m = 42000  # no of observations
n = 28000  # no of test cases
input_layer = 784
hidden_layer = 400
output_layer = 10
lamda = 1
y = zeros((m, 10))
X = zeros((m, 784))
test = zeros((n, 784))
test_output = zeros((n))
output_vector = zeros((m))
predict_vector = zeros((m))
theta = array((input_layer+1)*hidden_layer+(hidden_layer+1)*output_layer)


# function to randomly initialize the theta
def random_initialise(input, output):
    theta = random.random((output, input + 1)) * (0.07 * 2) - 0.07
    return theta


# function to read the .csv file
def read_training_set():
    file_read = zeros((m, 785))
    with open('train.csv') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            file_read[i] = row
            output_vector[i] = file_read[i][0]
            y[i][file_read[i][0]] = 1
            X[i] = file_read[i, 1:]
            i = i + 1


# function to read the test cases from test.csv file
def read_test_cases():
    file_read = zeros((n, 784))
    with open('test.csv') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            file_read[i] = row
            test[i] = file_read[i, :]
            i = i + 1


# function to calculate sigmoid
def sigmoid(X):
    temp = add(1, exp(multiply(X, -1)))
    if temp.any() == 0:
        print("temp ", temp)
    result = divide(1, temp)
    return result


# function to calculate the sigmoid gradient
def sigmoid_gradient(X):
    temp1 = exp(multiply(X, -1))
    temp2 = add(1, exp(multiply(X, -1))) ** 2
    # print("temp1 ", temp1)
    if temp2.any() == 0:
        print("temp2 ", temp2)
    result = divide(temp1, temp2)
    return result


# function to convert theta1 and theta2 into one vector
def roll_theta(theta1, theta2):
    return append(theta1.flatten(), theta2.flatten())


# function to return cost
def cost_function(theta, input_size, hidden_size, output_size, X, y, lamda):
    # unroll theta
    theta = nan_to_num(theta)
    theta1 = theta[:(input_size + 1) * hidden_size].reshape(hidden_size, input_size + 1)
    theta2 = theta[(input_size + 1) * hidden_size:].reshape(output_size, hidden_size + 1)

    # forward propagation
    A1 = insert(X, 0, 1, axis=1)
    Z2 = A1.dot(transpose(theta1))
    A2 = insert(sigmoid(Z2), 0, 1, axis=1)
    Z3 = A2.dot(transpose(theta2))
    A3 = sigmoid(Z3)

    # calculate cost
    J = 0
    J = J + (sum(log(A3) * y + (1 - y) * log(1 - A3))) / (-m) + (sum(theta1[:, 1:] ** 2) + sum(theta2[:, 1:] ** 2)) / (
        lamda * 2 * m)
    print("J = ", J)
    return J


# function to return gradient
def gradient_function(theta, input_size, hidden_size, output_size, X, y, lamda):
    # unroll theta
    theta = nan_to_num(theta)
    theta1 = theta[:(input_size + 1) * hidden_size].reshape(hidden_size, input_size + 1)
    theta2 = theta[(input_size + 1) * hidden_size:].reshape(output_size, hidden_size + 1)

    # forward propagation
    A1 = insert(X, 0, 1, axis=1)
    Z2 = A1.dot(transpose(theta1))
    A2 = insert(sigmoid(Z2), 0, 1, axis=1)
    Z3 = A2.dot(transpose(theta2))
    A3 = sigmoid(Z3)

    # calculate gradient
    delta_3 = A3 - y
    # delta_2 = ((delta_3.dot(theta2)) * sigmoid_gradient(insert(Z2, 0, 1, axis=1)))[:, 1:]
    delta_2 = ((delta_3.dot(theta2)) * A2 * (1 - A2))[:, 1:]
    temp_1 = copy(theta1)
    temp_2 = copy(theta2)
    temp_1[:, 0] = 0
    temp_2[:, 0] = 0


    theta2_grad = (transpose(delta_3).dot(A2)) / m + (lamda / m) * temp_2
    theta1_grad = (transpose(delta_2).dot(A1)) / m + (lamda / m) * temp_1
    theta1_grad = nan_to_num(theta1_grad)
    theta2_grad = nan_to_num(theta2_grad)
    # roll theta grad
    theta_grad = roll_theta(theta1_grad, theta2_grad)

    return theta_grad


# function to return the accuracy in percentage
def predict(theta):
    # unroll theta
    theta1 = theta[:(input_layer + 1) * hidden_layer].reshape(hidden_layer, input_layer + 1)
    theta2 = theta[(input_layer + 1) * hidden_layer:].reshape(output_layer, hidden_layer + 1)

    # forward propagation
    A1 = insert(X, 0, 1, axis=1)
    Z2 = A1.dot(transpose(theta1))
    A2 = insert(sigmoid(Z2), 0, 1, axis=1)
    Z3 = A2.dot(transpose(theta2))
    A3 = sigmoid(Z3)
    predict_vector = argmax(A3, axis=1)
    predict_vector = predict_vector - output_vector
    print("Accuracy = ", ((m - count_nonzero(predict_vector)) / m) * 100)
    return ((m - count_nonzero(predict_vector)) / m) * 100


# function to write the result in submission.csv
def write_output(theta, test):
    # unroll theta
    theta1 = theta[:(input_layer + 1) * hidden_layer].reshape(hidden_layer, input_layer + 1)
    theta2 = theta[(input_layer + 1) * hidden_layer:].reshape(output_layer, hidden_layer + 1)

    # forward propagation
    A1 = insert(test, 0, 1, axis=1)
    Z2 = A1.dot(transpose(theta1))
    A2 = insert(sigmoid(Z2), 0, 1, axis=1)
    Z3 = A2.dot(transpose(theta2))
    A3 = sigmoid(Z3)
    test_output = argmax(A3, axis=1)

    # writing output in the file
    with open('submission.csv','w') as testfile:
        fieldnames = ['ImageId', 'Label']
        writer = csv.DictWriter(testfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({'ImageId': i+1, 'Label': test_output[i]})
        print("Writting successful")








def decorated_cost(theta, input_layer, hidden_layer, output_layer, X, y, lamda):
    return cost_function(theta, input_layer, hidden_layer, output_layer, X, y, lamda)


def decorated_gradient(theta, input_layer, hidden_layer, output_layer, X, y, lamda):
    return gradient_function(theta, input_layer, hidden_layer, output_layer, X, y, lamda)


# function to write the parameters in a file
def write_function(theta):
    f = open("trained_parameters", 'w')
    for i in range(theta.shape[0]):
        print(theta[i], file=f)
    f.close()


# function to read the parameters from the file
def read_function(theta):
    f = open("trained_parameters").read()
    temp = f.split("\n")
    temp = temp[:-1]
    print(temp[0])
    temp = [float(i) for i in temp]
    print(temp[0])
    temp = [x*100000000 for x in temp]
    print(temp[0])
    theta = zeros(((input_layer+1)*hidden_layer +(hidden_layer+1)*output_layer))
    for i in range((input_layer+1)*hidden_layer +(hidden_layer+1)*output_layer):
        theta[i] = temp[i]
    print(theta)
    theta = divide(theta, 100000000, dtype = float128)
    print(theta)
    return theta




read_training_set()

theta1 = random_initialise(input_layer, hidden_layer)
theta2 = random_initialise(hidden_layer, output_layer)
theta = roll_theta(theta1, theta2)

#theta = read_function(theta)

args = (input_layer, hidden_layer, output_layer, X, y, lamda)
theta = optimize.fmin_cg(decorated_cost, x0=theta,
                         fprime=decorated_gradient,
                         args=args, maxiter=250, disp=True)

predict(theta)

write_function(theta)
cost_function(theta, input_layer, hidden_layer, output_layer, X, y, lamda)

read_test_cases()
write_output(theta, test)