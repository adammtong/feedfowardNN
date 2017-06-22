'''
Script Creates standard Feedforward Neural network from scratch                             
- Demonstrates forward and backpropagation on demo dataset from sklearn - make_moons              
  and itnroduces stochastic gradient descent and some tuning parameters such as learning rate,    
  regularisation rate and momentum           

    Author: Adam Ong                                                     
'''
import numpy as np
import sklearn.datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from sklearn.utils import shuffle
import timeit
from __future__ import division

# create dataset
np.random.seed(0)
X, y = sklearn.datasets.make_moons(10000, noise=0.20)

# Decision boundary for visualisation
def decision_boundary(proba):
    xx, yy = np.mgrid[-2:2:.01, -2:2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    if proba(grid).shape == xx.ravel().shape:
        probs = proba(grid).reshape(xx.shape)
    else:
        probs = proba(grid)[:,1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    ax.scatter(X[0:,0], X[0:, 1], c=y[0:], s=50,
           cmap=plt.cm.spectral, vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
    ax.set(aspect="equal",
       xlim=(-2, 2), ylim=(-2, 2),
       xlabel="$X_1$", ylabel="$X_2$")    

# Define and train Neural Net with SGD
def calculate_SGD(X ,y, num_hidden_neurons,mini_batch_size = 200, n_iter=100, learning_rate = 0.01, regularisation_rate = 0, momentum = 0.5, init = True):
    loss = [];
# initialise weights and biases
    X_size = len(X)
    np.random.seed(0)
    if init:
        W1 = np.random.randn(2, num_hidden_neurons) / np.sqrt(2)
        b1 = np.zeros((1, num_hidden_neurons))
        W2 = np.random.randn(num_hidden_neurons, 2) / np.sqrt(num_hidden_neurons)
        b2 = np.zeros((1, 2))
    else:
        W1 = np.ones((2, num_hidden_neurons))
        b1 = np.zeros((1, num_hidden_neurons))
        W2 = np.ones((num_hidden_neurons, 2))
        b2 = np.zeros((1, 2))
    model = {}
    # define weight updates
    W1_momentum=np.zeros((2,num_hidden_neurons))
    b1_momentum=np.zeros((1,num_hidden_neurons))
    W2_momentum=np.zeros((num_hidden_neurons,2))
    b2_momentum=np.zeros((1, 2))
    #iterate through epochs
    for i in xrange(n_iter):   
        X_copy, y_copy = shuffle(X, y, random_state=0)
        mini_batches = [X_copy[k:k+mini_batch_size] for k in xrange(0,X_size,mini_batch_size)]
        y_batches = [y_copy[k:k+mini_batch_size] for k in xrange(0, X_size,mini_batch_size)]
        j=0
        print(i)
        #iterate through mini_batches
        for batch, y_batch in zip(mini_batches,y_batches):
            if i == 150:
                learning_rate = learning_rate/10
                print("learning rate: {}".format(learning_rate))
        #feedforward
            X_batch= batch
            Z1 = X_batch.dot(W1)+b1
            a1 = np.tanh(Z1)
            Z2 = a1.dot(W2)+b2
            yhat = softmax(Z2)
        #backpropagation
            d2 = yhat
            d2[range(len(batch)),y_batch] -= 1
            d1 = (1-a1**2)*d2.dot(W2.T)
            dW2 = a1.T.dot(d2)
            db2 = np.sum(d2, axis=0, keepdims = True)
            dW1 = X_batch.T.dot(d1)
            db1 = np.sum(d1, axis=0)
        #regularisation
            dW1 += regularisation_rate * W1     
            dW2 += regularisation_rate * W2  
            
            W1_momentum = momentum * W1_momentum-learning_rate * dW1
            b1_momentum = momentum * b1_momentum-learning_rate * db1
            W2_momentum = momentum * W2_momentum-learning_rate * dW2
            b2_momentum = momentum * b2_momentum-learning_rate * db2
        #gradient descent       
            W1 = W1 + W1_momentum
            b1 = b1 + b1_momentum
            W2 = W2 + W2_momentum
            b2 = b2 + b2_momentum
            
            model = {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2}
            curr_loss = predict(model,X,y,True)
            print("mini_batch {} of epoch {} - current loss: {}".format(j,i,curr_loss))
            loss.append(curr_loss)
            j+=1
    return loss, model

# Predict y_hat from input X or calculate loss
def predict(model, X, y, calc_loss = False):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    Z1 = X.dot(W1) + b1
    a1 = np.tanh(Z1)
    Z2 = a1.dot(W2) + b2
    probs = softmax(Z2)
    if calc_loss:
        loss = -np.log(probs[range(len(X)), y])
        tot_loss = np.sum(loss)
        return 1./len(X) * tot_loss
    return np.argmax(probs, axis=1) 
 
# Compute softmax values for each sets of scores in x.
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims = True)
    
#function call - 10 neurons in the hidden layer
start_time = timeit.default_timer()
loss_SGD, model_SGD = calculate_SGD(X, y, 10, mini_batch_size = 128, n_iter = 5, learning_rate = 0.003, regularisation_rate = 0.001, momentum=0.9)
elapsed_SGD = timeit.default_timer() - start_time

# plot decision boundary
decision_boundary(lambda x: predict(model_SGD,x,y))

# plot loss
plt.axis([0,len(loss_SGD),0,0.6])
plt.plot(loss_SGD, label = "sgd")
plt.title("batch vs sgd loss")
plt.xlabel("number of minibatches evaluated")
plt.ylabel("loss")
plt.legend()
plt.show()


