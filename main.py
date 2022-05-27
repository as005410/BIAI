#import the necessary packages
from distutils.log import error
from tkinter import W
import numpy as np


class NeuralNetwork:
	#constructor of NN
	#layers->characteristic of layers
	#alpha->learning rate
    def __init__(self, layers, alpha=0.1):
      # initialize the list of weight matrices
      self.W = []
      #number of layers
      self.layers = layers
      #learning rate
      self.alpha = alpha 
      #loop through layers
      for i in np.arange(0, len(layers)-2):
        w = np.random.randn(layers[i] + 1,layers[i + 1] + 1)
        self.W.append(w / np.sqrt(layers[i]))
      # connections need a bias term but the output does not
      w = np.random.randn(layers[-2] + 1, layers[-1])
      self.W.append(w / np.sqrt(layers[-2]))

    # construct and return a string that represents the network acrhitecture
    def __repr__(self):
      return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    #compute sigmoid activation function
    def sigmoid(self, x):
      return 1.0/(1+np.exp(-x))

    # compute the derivative
    def sigmoid_deriv(self, x):
      return x*(1-x)

    #return activations
    def fit_partial(self, x, y):
      
      #feedforward
      A = [np.atleast_2d(x)]
      #loop through layers in network
      for layer in np.arange(0, len(self,W)):
        net = A[layer].dot(self.W[layer])
        out = self.sigmoid(net)
        A.append(out)

      #backpropagation
      error = A[-1] - y
      D = [error * self.sigmoid_deriv(A[-1])]

      for layer in np.arange(len(A) - 2, 0, -1) :
        delta = D[-1].dot(self.W[layer].T)
        delta = delta * self.sigmoid_deriv(A[layer])
        D.append(delta)

      D = D[::-1]

      for layer in np.arange(0, len(self.W)):
        self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])


    #function for training
    #X=training data
    #y=label
    def fit(self, X, y, epochs=1000, displayUpdate=100):
      
      X = np.c_[X, np.ones((X.shape[0]))]
		
      # loop over the epochs and traiin network
      for epoch in np.arange(0, epochs):
        for (x,target) in zip(X,y):
          self.fit_partial(x, target)

        if epoch == 0 or (epoch+1) % displayUpdate == 0:
          loss =  self.calculate_loss(X,y)
          print("epoch = {}, loss = {:.7f}".format(epoch+1,loss))

    def predict(self, X, addBias=True):
      p = np.atleast_2d(X)
      if addBias:
        p = np.c_[p, np.ones((p.shape[0]))]
      
      for layer in np.arange(0, len(self.W)):
        p = self.sigmoid(np.dot(p, self.W[layer]))

      return p


    def calculate_loss(self, X, targets):
      targets = np.atleast_2d(targets)
      predictions = self.predict(X, addBias=False)
      loss = 0.5 * np.sum((predictions - targets) ** 2)

      return loss
